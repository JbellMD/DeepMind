# DeepMind Chatbot Platform

A commercial-grade chatbot platform built with DeepSeek's fine-tuned LLM, optimized for M3 Studio with 512GB unified memory.

## Overview

This project implements a high-performance conversational AI system with a modern, responsive UI similar to leading commercial AI platforms. The architecture is designed for optimal performance on Apple Silicon, featuring DeepSeek's fine-tuned model with vLLM/DeepSpeed acceleration.

## Key Features

- ✅ **High-Performance Inference**: Optimized for Apple Silicon with MPS/Metal support
- ✅ **Modern UI**: React/Next.js frontend with Tailwind CSS and Framer Motion animations
- ✅ **Streaming Responses**: Real-time token streaming via Server-Sent Events
- ✅ **Context Management**: Efficient handling of conversation history
- ✅ **Optional RAG Support**: Vector database integration for retrieval-augmented generation
- ✅ **Advanced Capabilities**: Function calling, tool use, and plugin system

## Architecture

```
Frontend (Next.js + Tailwind) → API Gateway (FastAPI) → Inference Engine (DeepSeek + vLLM) → Optional Storage (Vector DB + Postgres)
```

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- CUDA-compatible GPU or Apple Silicon M-series chip

### Installation

1. Clone the repository
2. Set up the backend:
   ```bash
   cd backend
   pip install -r requirements.txt
   python main.py
   ```
3. Set up the frontend:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## Development

### Backend

The backend uses FastAPI with optimized inference via vLLM or DeepSpeed, supporting:
- Token streaming
- FP16/INT4 quantization
- KV caching
- Prompt templates
- Tool/function calling

### Frontend

The frontend uses Next.js with Tailwind CSS and implements:
- Modern chat interface
- Real-time typing animations
- Markdown rendering
- Code syntax highlighting
- Dark/light mode
- Responsive design

## Deployment

Docker configurations are provided for easy deployment:
- Backend container with GPU support
- Frontend container
- Optional database containers

## License

[MIT License](LICENSE)
