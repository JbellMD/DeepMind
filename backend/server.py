#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import asyncio
from typing import List, Dict, Any, Optional, Union, AsyncIterator

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# Configuration
MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-instruct"
DEFAULT_SYSTEM_PROMPT = """You are DeepMind AI, an advanced AI assistant built on a fine-tuned DeepSeek model.
You are designed to be helpful, harmless, and honest in all your interactions.
You excel at providing detailed, thoughtful responses while maintaining a friendly and conversational tone.
"""

# Global model and tokenizer instances
model = None
tokenizer = None

# Initialize FastAPI app
app = FastAPI(title="DeepMind AI API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    system_prompt: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    message: Message
    created: int = Field(default_factory=lambda: int(time.time()))

async def load_model():
    """Lazily load the model and tokenizer."""
    global model, tokenizer
    
    if model is not None and tokenizer is not None:
        return model, tokenizer
        
    try:
        logger.info(f"Loading model: {MODEL_NAME}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        
        # Load model with optimizations
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,  # Use half precision
            trust_remote_code=True,
            device_map="auto",
            load_in_8bit=True if torch.cuda.is_available() else False,
        )
        
        logger.info(f"Model {MODEL_NAME} loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")

def format_prompt(messages: List[Message], system_prompt: Optional[str] = None) -> str:
    """Format the conversation history into a prompt for the model."""
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    
    # Create the conversation list with system prompt
    conversation = [{"role": "system", "content": system_prompt}]
    
    # Add the user messages
    for message in messages:
        role = message.role.lower()
        content = message.content
        
        if role in ["user", "assistant", "system"]:
            conversation.append({"role": role, "content": content})
    
    # Use the tokenizer's chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
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
