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
from rag_service import RAGService
from vector_store import VectorStore
from document_processor import DocumentProcessor, TextSplitter
from embedding_service import EmbeddingService

# Global model and tokenizer instances
MODEL = None
TOKENIZER = None
RAG_SERVICE = None

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

async def load_rag_service():
    """Lazily load the RAG service."""
    global RAG_SERVICE
    
    if RAG_SERVICE is not None:
        return RAG_SERVICE
    
    try:
        # Initialize vector store
        vector_store = VectorStore(data_dir="data/vector_store")
        
        # Initialize document processor
        text_splitter = TextSplitter(chunk_size=1000, chunk_overlap=200)
        document_processor = DocumentProcessor(text_splitter=text_splitter)
        
        # Initialize embedding service
        embedding_service = EmbeddingService(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Initialize RAG service
        RAG_SERVICE = RAGService(
            vector_store=vector_store,
            document_processor=document_processor,
            embedding_service=embedding_service,
            max_context_documents=5,
            similarity_threshold=0.6
        )
        
        logger.info("RAG service loaded successfully")
        return RAG_SERVICE
    
    except Exception as e:
        logger.error(f"Failed to load RAG service: {str(e)}")
        raise RuntimeError(f"Failed to load RAG service: {str(e)}")

def format_prompt(messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
    """Format the conversation history into a prompt for the model.
    
    This specifically formats for DeepSeek's chat template. Different models may require
    different formatting approaches.
    """
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    
    conversation = []
    
    # Add the system prompt
    conversation.append({"role": "system", "content": system_prompt})
    
    # Add messages
    for message in messages:
        role = message.get("role", "").lower()
        content = message.get("content", "")
        
        # Only include valid message types
        if role in ["user", "assistant", "system"]:
            conversation.append({"role": role, "content": content})
    
    # Use the tokenizer's chat template if available
    if hasattr(TOKENIZER, "apply_chat_template"):
        formatted_prompt = TOKENIZER.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=True
        )
    else:
        # Fallback to manual formatting
        formatted_prompt = ""
        for msg in conversation:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted_prompt += f"<|system|>\n{content}\n"
            elif role == "user":
                formatted_prompt += f"
