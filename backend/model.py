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
                formatted_prompt += f"<|user|>\n{content}\n"
            elif role == "assistant":
                formatted_prompt += f"<|assistant|>\n{content}\n"
    
    return formatted_prompt

async def generate_response(messages: List[Dict[str, str]], max_new_tokens: int = 150, temperature: float = 0.7) -> str:
    """
    Generate a response from the model given a conversation history.

    Args:
        messages (List[Dict[str, str]]): The conversation history.
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.

    Returns:
        str: The generated response.
    """
    model, tokenizer = await load_model()
    prompt = format_prompt(messages)
    logger.info("Generating response with prompt:")
    logger.info(prompt)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate output tokens
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode generated tokens
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Post-process to remove the prompt part if necessary (assuming the prompt is at the start)
    if response_text.startswith(prompt):
        response_text = response_text[len(prompt):]
    
    return response_text.strip()

async def stream_response(messages: List[Dict[str, str]], max_new_tokens: int = 150, temperature: float = 0.7) -> AsyncIterator[str]:
    """
    Asynchronously stream a generated response from the model given a conversation history.
    
    This function yields parts of the generated text as they become available.
    
    Args:
        messages (List[Dict[str, str]]): The conversation history.
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.
        
    Yields:
        str: A chunk of the generated response.
    """
    model, tokenizer = await load_model()
    prompt = format_prompt(messages)
    logger.info("Streaming response with prompt:")
    logger.info(prompt)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Use model.generate with output_scores and return_dict_in_generate to simulate streaming
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        output_scores=True,
        return_dict_in_generate=True,
    )
    
    generated_ids = output.sequences[0]
    
    # Simple token-by-token streaming simulation
    for token_id in generated_ids[len(inputs["input_ids"][0]):]:
        token = tokenizer.decode(token_id.unsqueeze(0), skip_special_tokens=True)
        yield token
        await asyncio.sleep(0.05)  # simulate streaming delay

# Example main function to test generation
if __name__ == "__main__":
    async def main():
        test_messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm good, thank you! How can I help you today?"}
        ]
        # Test synchronous generation
        response = await generate_response(test_messages)
        print("Generated Response:")
        print(response)
        
        # Test streaming response
        print("Streaming Response:")
        async for token in stream_response(test_messages):
            print(token, end="", flush=True)
    
    asyncio.run(main())

                
