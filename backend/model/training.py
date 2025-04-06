#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Optional, Dict, Any
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)
import torch.distributed as dist
from loguru import logger

from .optimization import (
    get_optimal_batch_size,
    setup_memory_optimization,
    monitor_memory_usage
)

class ModelTrainer:
    """Handles model fine-tuning with optimizations for Apple Silicon."""
    
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        use_mps: bool = True,
        use_peft: bool = True,
        lora_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the trainer.
        
        Args:
            model_name: Name or path of the base model
            output_dir: Directory to save fine-tuned model
            use_mps: Whether to use MPS (Metal Performance Shaders)
            use_peft: Whether to use PEFT/LoRA for fine-tuning
            lora_config: Configuration for LoRA if use_peft is True
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = "mps" if use_mps and torch.backends.mps.is_available() else "cpu"
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with optimizations
        self.model = self._load_model(use_peft, lora_config)
        
        logger.info(f"Initialized trainer with model {model_name} on {self.device}")
    
    def _load_model(self, use_peft: bool, lora_config: Optional[Dict[str, Any]]):
        """Load and optimize the model for training."""
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        
        # Apply memory optimizations
        setup_memory_optimization(model)
        
        if use_peft:
            # Default LoRA configuration
            default_lora_config = {
                "r": 8,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj"],
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": "CAUSAL_LM"
            }
            
            # Update with provided config
            if lora_config:
                default_lora_config.update(lora_config)
            
            # Prepare model for PEFT
            model = prepare_model_for_kbit_training(model)
            config = LoraConfig(**default_lora_config)
            model = get_peft_model(model, config)
            
            logger.info("Applied LoRA adaptation to model")
        
        return model
    
    def train(
        self,
        train_dataset,
        eval_dataset=None,
        batch_size: Optional[int] = None,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 0.3,
        warmup_ratio: float = 0.03
    ):
        """Fine-tune the model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            batch_size: Batch size (if None, will be automatically optimized)
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            gradient_accumulation_steps: Number of steps for gradient accumulation
            max_grad_norm: Maximum gradient norm for clipping
            warmup_ratio: Ratio of warmup steps
        """
        # Get optimal batch size if not provided
        if batch_size is None:
            batch_size = get_optimal_batch_size()
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            warmup_ratio=warmup_ratio,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch" if eval_dataset else "no",
            fp16=True,
            report_to="tensorboard"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer
        )
        
        # Add memory monitoring
        trainer.add_callback(monitor_memory_usage())
        
        # Start training
        logger.info("Starting model fine-tuning")
        trainer.train()
        
        # Save the final model
        trainer.save_model()
        logger.info(f"Model saved to {self.output_dir}")
    
    def export_model(self, output_path: Optional[str] = None):
        """Export the fine-tuned model.
        
        Args:
            output_path: Path to save the model (if None, uses output_dir)
        """
        output_path = output_path or self.output_dir
        
        # Save model and tokenizer
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        logger.info(f"Model exported to {output_path}")
