#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import time
import asyncio
from typing import List, Dict, Any, Optional, Union

import torch
from fastapi import FastAPI, Request, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logger
logger.add("logs/api.log", rotation="10 MB", level="INFO")

# Initialize FastAPI app
app = FastAPI(
    title="DeepMind Chatbot API",
    description="High-performance API for DeepSeek-powered conversational AI",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, ge=1, le=8192)
    stream: bool = True
    system_prompt: Optional[str] = None

class ChatResponse(BaseModel):
    message: Message
    created: int
    model: str

# Initialize model (will be lazily loaded when needed)
MODEL = None
TOKENIZER = None

def get_device():
    """Determine the optimal device for inference."""
    if torch.backends.mps.is_available():
        # Use Metal Performance Shaders for Apple Silicon
        return torch.device("mps")
    elif torch.cuda.is_available():
        # Use CUDA for NVIDIA GPUs
        return torch.device("cuda")
    else:
        # Fallback to CPU
        return torch.device("cpu")

async def load_model():
    """Lazily load the model and tokenizer."""
    global MODEL, TOKENIZER
    
    if MODEL is not None and TOKENIZER is not None:
        return MODEL, TOKENIZER
        
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Change to your fine-tuned model path or HF model ID
        model_name = os.getenv("MODEL_PATH", "deepseek-ai/deepseek-coder-33b-instruct")
        
        logger.info(f"Loading model: {model_name}")
        device = get_device()
        logger.info(f"Using device: {device}")
        
        # Load tokenizer
        TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with optimizations
        MODEL = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use half precision
            trust_remote_code=True,
            device_map="auto",
            load_in_8bit=True if device.type == "cuda" else False,
        )
        
        logger.info("Model loaded successfully")
        return MODEL, TOKENIZER
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")

def format_prompt(messages: List[Message], system_prompt: Optional[str] = None) -> str:
    """Format the conversation history into a prompt for the model."""
    formatted_prompt = ""
    
    # Add system prompt if provided
    if system_prompt:
        formatted_prompt += f"<|system|>\n{system_prompt}\n"
    
    # Add conversation history
    for message in messages:
        role = message.role.lower()
        if role == "user":
            formatted_prompt += f"
