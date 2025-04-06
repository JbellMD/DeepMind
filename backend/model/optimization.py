#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gc
import psutil
import torch
from typing import Optional
from loguru import logger
from transformers import TrainerCallback
import torch.nn as nn

# Constants for memory management
MAX_BATCH_SIZE = 32
MIN_BATCH_SIZE = 1
MEMORY_BUFFER = 0.2  # Keep 20% memory free
MODEL_MEMORY_FOOTPRINT = 2.5  # GB per batch item (approximate)

def get_available_memory():
    """Get available system memory in GB."""
    if torch.backends.mps.is_available():
        # For Apple Silicon, use system memory
        memory = psutil.virtual_memory()
        return memory.available / (1024 ** 3)  # Convert to GB
    else:
        # For CUDA devices
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return psutil.virtual_memory().available / (1024 ** 3)

def get_optimal_batch_size(
    available_memory: Optional[float] = None,
    max_batch: int = MAX_BATCH_SIZE,
    min_batch: int = MIN_BATCH_SIZE
) -> int:
    """Calculate optimal batch size based on available memory.
    
    Args:
        available_memory: Available memory in GB (if None, will be detected)
        max_batch: Maximum allowed batch size
        min_batch: Minimum allowed batch size
        
    Returns:
        Optimal batch size
    """
    if available_memory is None:
        available_memory = get_available_memory()
    
    # Calculate optimal batch size
    optimal_size = int(available_memory * (1 - MEMORY_BUFFER) / MODEL_MEMORY_FOOTPRINT)
    
    # Clamp between min and max
    return min(max_batch, max(min_batch, optimal_size))

def setup_memory_optimization(model: nn.Module):
    """Apply memory optimizations to the model.
    
    Args:
        model: The model to optimize
    """
    # Enable gradient checkpointing
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    # Enable input requires grad
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    
    # Use efficient attention if available
    if hasattr(model, "config"):
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Force garbage collection
    gc.collect()
    
    logger.info("Applied memory optimizations to model")

class MemoryTracker(TrainerCallback):
    """Callback to track memory usage during training."""
    
    def __init__(self, log_interval: int = 100):
        """Initialize the memory tracker.
        
        Args:
            log_interval: Number of steps between memory logs
        """
        self.log_interval = log_interval
    
    def _log_memory(self, args, state, prefix: str = ""):
        """Log current memory usage."""
        if torch.backends.mps.is_available():
            # For Apple Silicon
            memory = psutil.virtual_memory()
            used_gb = (memory.total - memory.available) / (1024 ** 3)
            total_gb = memory.total / (1024 ** 3)
            logger.info(f"{prefix}Memory Usage: {used_gb:.1f}GB / {total_gb:.1f}GB")
        elif torch.cuda.is_available():
            # For CUDA devices
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            logger.info(f"{prefix}GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
    
    def on_step_end(self, args, state, control):
        """Called at the end of each step."""
        if state.global_step % self.log_interval == 0:
            self._log_memory(args, state, "Step End - ")
    
    def on_evaluate(self, args, state, control):
        """Called when evaluation starts."""
        self._log_memory(args, state, "Evaluation - ")
    
    def on_save(self, args, state, control):
        """Called when model is saved."""
        self._log_memory(args, state, "Save - ")

def monitor_memory_usage(log_interval: int = 100) -> MemoryTracker:
    """Create a memory usage monitor.
    
    Args:
        log_interval: Number of steps between memory logs
        
    Returns:
        MemoryTracker callback
    """
    return MemoryTracker(log_interval)

def optimize_inference_settings(
    model: nn.Module,
    batch_size: Optional[int] = None,
    use_cache: bool = True
):
    """Optimize model settings for inference.
    
    Args:
        model: The model to optimize
        batch_size: Batch size (if None, will be automatically determined)
        use_cache: Whether to use KV cache during inference
    """
    # Set optimal batch size
    if batch_size is None:
        batch_size = get_optimal_batch_size()
    
    # Configure model settings
    if hasattr(model, "config"):
        model.config.use_cache = use_cache
    
    # Apply memory optimizations
    setup_memory_optimization(model)
    
    # Set evaluation mode
    model.eval()
    
    logger.info(f"Model optimized for inference with batch size {batch_size}")
    return batch_size

def cleanup_memory():
    """Perform memory cleanup operations."""
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Force garbage collection
    gc.collect()
    
    # Log memory status
    if torch.backends.mps.is_available():
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024 ** 3)
        logger.info(f"Memory cleanup completed. Available: {available_gb:.1f}GB")
