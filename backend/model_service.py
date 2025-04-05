#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import asyncio
from typing import List, Dict, Any, Optional, Union, AsyncIterator, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger

from config import MODEL_CONFIG, get_device, DEFAULT_SYSTEM_PROMPT

# Global model and tokenizer instances
MODEL = None
TOKENIZER = None

async def load_model():
    """Lazily load the model and tokenizer."""
    global MODEL, TOKENIZER
    
    if MODEL is not None and TOKENIZER is not None:
        return MODEL, TOKENIZER
        
    try:
        model_name = MODEL_CONFIG["model_name"]
        logger.info(f"Loading model: {model_name}")
        device = get_device()
        
        # Load tokenizer
        TOKENIZER = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Load model with optimizations
        MODEL = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use half precision
            trust_remote_code=True,
            device_map="auto",
            load_in_8bit=True if torch.cuda.is_available() else False,
        )
        
        logger.info(f"Model {model_name} loaded successfully")
        return MODEL, TOKENIZER
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")

def format_chat_prompt(messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
    """Format the conversation history for DeepSeek's chat models."""
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    
    # Create the conversation list with system prompt
    conversation = [{"role": "system", "content": system_prompt}]
    
    # Add the user messages
    for message in messages:
        role = message.get("role", "").lower()
        content = message.get("content", "")
        
        if role in ["user", "assistant", "system"]:
            conversation.append({"role": role, "content": content})
    
    # Let the tokenizer handle the formatting if it supports it
    if TOKENIZER and hasattr(TOKENIZER, "apply_chat_template"):
        return TOKENIZER.apply_chat_template(
            conversation, 
            tokenize=False,
            add_generation_prompt=True
        )
    
    # Fallback manual formatting for DeepSeek models
    formatted_prompt = ""
    
    for msg in conversation:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            formatted_prompt += f"<|system|>\n{content}\n"
        elif role == "user":
            formatted_prompt += f"
