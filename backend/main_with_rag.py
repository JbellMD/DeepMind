#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import time
import asyncio
from typing import List, Dict, Any, Optional, Union, AsyncIterator

import torch
import uvicorn
from fastapi import FastAPI, Request, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from loguru import logger
from dotenv import load_dotenv

# Import RAG components (assuming these are custom modules you’ve defined)
from rag_api import router as rag_router
from rag_service import RAGService
from vector_store import VectorStore
from document_processor import DocumentProcessor
from embedding_service import EmbeddingService

# Load environment variables
load_dotenv()

# Configure logger
os.makedirs("logs", exist_ok=True)
logger.add("logs/api.log", rotation="10 MB", level="INFO")

# Initialize FastAPI app
app = FastAPI(
    title="DeepMind Chatbot API",
    description="High-performance API for DeepSeek-powered conversational AI with RAG capabilities",
    version="1.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include RAG router
app.include_router(rag_router)

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
    use_rag: bool = False  # New parameter to enable/disable RAG

class ChatResponse(BaseModel):
    message: Message
    created: int
    model: str
    sources: Optional[List[Dict[str, Any]]] = None  # Sources from RAG if enabled

# Initialize model (will be lazily loaded when needed)
MODEL = None
TOKENIZER = None
MODEL_NAME = os.getenv("MODEL_PATH", "deepseek-ai/deepseek-coder-33b-instruct")

# Initialize RAG service (will be lazily loaded when needed)
RAG_SERVICE = None
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "data/vector_store")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

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
        
        logger.info(f"Loading model: {MODEL_NAME}")
        device = get_device()
        logger.info(f"Using device: {device}")
        
        # Load tokenizer
        TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Load model with optimizations
        MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
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

async def load_rag_service():
    """Lazily load the RAG service."""
    global RAG_SERVICE
    
    if RAG_SERVICE is not None:
        return RAG_SERVICE
    
    try:
        # Initialize vector store
        vector_store = VectorStore(data_dir=VECTOR_STORE_DIR)
        
        # Initialize document processor
        document_processor = DocumentProcessor()
        
        # Initialize embedding service
        embedding_service = EmbeddingService(model_name=EMBEDDING_MODEL)
        
        # Initialize RAG service
        RAG_SERVICE = RAGService(
            vector_store=vector_store,
            document_processor=document_processor,
            embedding_service=embedding_service
        )
        
        logger.info("RAG service loaded successfully")
        return RAG_SERVICE
    
    except Exception as e:
        logger.error(f"Failed to load RAG service: {str(e)}")
        raise RuntimeError(f"Failed to load RAG service: {str(e)}")

def format_prompt(messages: List[Message], system_prompt: Optional[str] = None) -> str:
    """Format the conversation history into a prompt for the model."""
    formatted_prompt = ""
    
    # Add system prompt if provided
    if system_prompt:
        formatted_prompt += f"<|system|>\n{system_prompt}\n<|end|>\n"
    
    # Add conversation history
    for message in messages:
        role = message.role.lower()
        if role == "user":
            formatted_prompt += f"<|user|>\n{message.content}\n<|end|>\n"
        elif role == "assistant":
            formatted_prompt += f"<|assistant|>\n{message.content}\n<|end|>\n"
        else:
            logger.warning(f"Unknown role '{role}' in message, skipping")
    
    # Add assistant start token if needed
    formatted_prompt += "<|assistant|>\n"
    return formatted_prompt

async def generate_stream(prompt: str, temperature: float, max_tokens: int) -> AsyncIterator[str]:
    """Generate a streaming response from the model."""
    model, tokenizer = await load_model()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    async for token in model.generate(
        **inputs,
        max_length=max_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        stream=True  # Hypothetical streaming support; adjust based on actual model API
    ):
        yield tokenizer.decode(token, skip_special_tokens=True)

@app.post("/chat", response_model=None)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks
) -> Union[EventSourceResponse, ChatResponse]:
    """Handle chat requests with streaming or non-streaming responses."""
    try:
        # Load model and tokenizer
        model, tokenizer = await load_model()
        
        # Format the prompt
        prompt = format_prompt(request.messages, request.system_prompt)
        
        # Handle RAG if enabled
        sources = None
        if request.use_rag:
            rag_service = await load_rag_service()
            last_user_message = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
            if last_user_message:
                retrieved_docs = rag_service.retrieve(last_user_message, top_k=3)
                context = "\n".join([doc["content"] for doc in retrieved_docs])
                prompt += f"\n<|context|>\n{context}\n<|end|>\n"
                sources = retrieved_docs

        if request.stream:
            # Streaming response using Server-Sent Events (SSE)
            async def event_generator():
                start_time = int(time.time())
                async for chunk in generate_stream(prompt, request.temperature, request.max_tokens):
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "message": {"role": "assistant", "content": chunk},
                            "created": start_time,
                            "model": MODEL_NAME,
                            "sources": sources
                        })
                    }
                yield {"event": "done", "data": ""}

            return EventSourceResponse(event_generator())
        else:
            # Non-streaming response
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_length=request.max_tokens,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return ChatResponse(
                message=Message(role="assistant", content=response_text),
                created=int(time.time()),
                model=MODEL_NAME,
                sources=sources
            )
    
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Run the app
if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=host, port=port)