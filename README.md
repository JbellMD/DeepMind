# DeepMind AI Chat Platform

A commercial-grade chatbot platform built with DeepSeek's fine-tuned LLM, optimized for M3 Studio with 512GB unified memory.

## Overview

This project implements a high-performance conversational AI system with a modern, responsive UI similar to leading commercial AI platforms. The architecture is designed for optimal performance on Apple Silicon, featuring DeepSeek's fine-tuned model with vLLM/DeepSpeed acceleration.

## Key Features

- **High-Performance Inference**: Optimized for Apple Silicon with MPS/Metal support
- **Modern UI**: React/Next.js frontend with Tailwind CSS and Framer Motion animations
- **Streaming Responses**: Real-time token streaming via Server-Sent Events
- **Context Management**: Efficient handling of conversation history
- **Optional RAG Support**: Vector database integration for retrieval-augmented generation
- **Advanced Capabilities**: Function calling, tool use, and plugin system

## Project Structure

```
DeepMind/
│
├── backend/               # FastAPI backend for model serving
│   ├── config.py          # Configuration settings
│   ├── api.py             # API endpoints
│   ├── model_service.py   # Model management
│   ├── server.py          # Server startup script
│   └── requirements.txt   # Python dependencies
│
├── frontend/              # Next.js frontend
│   ├── components/        # React components
│   ├── pages/             # Next.js pages
│   ├── styles/            # CSS and styling
│   ├── utils/             # Utility functions
│   └── public/            # Static assets
│
├── shared/                # Shared resources
├── docker/                # Docker configurations
└── README.md              # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- CUDA-compatible GPU or Apple Silicon M-series chip

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the FastAPI server:
   ```bash
   python server.py
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open your browser to [http://localhost:3000](http://localhost:3000)

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

## Configuration

Create a `.env` file in the backend directory with the following variables:

```
MODEL_PATH=your-fine-tuned-deepseek-model
MAX_TOKENS=4096
TEMPERATURE=0.7
TOP_P=0.9
TOP_K=50
HOST=0.0.0.0
PORT=8000
DEBUG=False
```

## Deployment

Docker configurations are provided for easy deployment:
- Backend container with GPU support
- Frontend container
- Optional database containers

## License

[MIT License](LICENSE)
