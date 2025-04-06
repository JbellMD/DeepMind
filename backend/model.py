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
                formatted_prompt += f"<|system|>\n{content}\n<|end|>\n"
            elif role == "user":
                formatted_prompt += f"<|user|>\n{content}\n<|end|>\n"
            elif role == "assistant":
                formatted_prompt += f"<|assistant|>\n{content}\n<|end|>\n"
        formatted_prompt += "<|assistant|>\n"  # Start assistant response
    
    return formatted_prompt

async def generate_response(
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    use_rag: bool = False,
    stream: bool = False
) -> Union[str, AsyncIterator[str]]:
    """Generate a response from the model, with optional RAG and streaming."""
    model, tokenizer = await load_model()
    
    # Format the base prompt
    prompt = format_prompt(messages, system_prompt)
    
    # Integrate RAG if enabled
    if use_rag:
        rag_service = await load_rag_service()
        last_user_message = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        if last_user_message:
            retrieved_docs = rag_service.retrieve(last_user_message, top_k=3)
            context = "\n".join([doc["content"] for doc in retrieved_docs])
            prompt += f"\n<|context|>\n{context}\n<|end|>\n"
            logger.info(f"RAG context added: {len(retrieved_docs)} documents retrieved")
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    if stream:
        # Streaming response
        async def stream_generator():
            for output in model.generate(
                **inputs,
                max_length=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                # Note: Streaming is simulated here; adjust if model supports native streaming
            ):
                yield tokenizer.decode(output[0], skip_special_tokens=True)
        
        return stream_generator()
    else:
        # Non-streaming response
        outputs = model.generate(
            **inputs,
            max_length=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

async def main():
    """Simple CLI for testing the chatbot."""
    logger.info("Starting DeepMind Chatbot CLI")
    await load_model()  # Preload model
    await load_rag_service()  # Preload RAG service
    
    conversation = []
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                break
            
            # Add user message to conversation
            conversation.append({"role": "user", "content": user_input})
            
            # Generate response
            start_time = time.time()
            response = await generate_response(
                messages=conversation,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                temperature=0.7,
                max_tokens=1024,
                use_rag=True,  # Toggle RAG here
                stream=False
            )
            
            print(f"Bot: {response}")
            logger.info(f"Response generated in {time.time() - start_time:.2f} seconds")
            
            # Add assistant response to conversation
            conversation.append({"role": "assistant", "content": response})
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error in CLI: {str(e)}")
            print("An error occurred. Please try again.")

if __name__ == "__main__":
    # Configure logger
    logger.add("logs/chatbot.log", rotation="10 MB", level="INFO")
    asyncio.run(main())