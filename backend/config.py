#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()

# Configure logger
os.makedirs("logs", exist_ok=True)
logger.add("logs/api.log", rotation="10 MB", level="INFO")

# Model configuration
MODEL_CONFIG = {
    "model_name": os.getenv("MODEL_PATH", "deepseek-ai/deepseek-coder-33b-instruct"),
    "max_tokens": int(os.getenv("MAX_TOKENS", "4096")),
    "temperature": float(os.getenv("TEMPERATURE", "0.7")),
    "top_p": float(os.getenv("TOP_P", "0.9")),
    "top_k": int(os.getenv("TOP_K", "50")),
    "repetition_penalty": float(os.getenv("REPETITION_PENALTY", "1.1")),
}

# API configuration
API_CONFIG = {
    "host": os.getenv("HOST", "0.0.0.0"),
    "port": int(os.getenv("PORT", "8000")),
    "debug": os.getenv("DEBUG", "False").lower() == "true",
}

# Get default system prompt
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "DEFAULT_SYSTEM_PROMPT",
    "You are DeepMind, an AI assistant built by our organization. You're designed to be helpful, harmless, and honest in all interactions."
)

def get_device():
    """Determine the optimal device for inference."""
    if torch.backends.mps.is_available():
        # Use Metal Performance Shaders for Apple Silicon
        logger.info("Using MPS (Metal Performance Shaders) for Apple Silicon")
        return torch.device("mps")
    elif torch.cuda.is_available():
        # Use CUDA for NVIDIA GPUs
        logger.info(f"Using CUDA with {torch.cuda.device_count()} GPU(s)")
        return torch.device("cuda")
    else:
        # Fallback to CPU
        logger.info("No GPU detected, using CPU for inference (this will be slow)")
        return torch.device("cpu")
