# DeepMind AI Chat Platform

A commercial-grade chatbot platform built with DeepSeek's fine-tuned LLM, optimized for M3 Studio with 512GB unified memory.

## Overview

This project implements a high-performance conversational AI system with a modern, responsive UI similar to leading commercial AI platforms. The architecture is designed for optimal performance on Apple Silicon, featuring DeepSeek's fine-tuned model with vLLM/DeepSpeed acceleration.

## Key Features

- **High-Performance Inference**: Optimized for Apple Silicon with MPS/Metal support
- **Modern UI**: React/Next.js frontend with Tailwind CSS and Framer Motion animations
- **Streaming Responses**: Real-time token streaming via Server-Sent Events
- **Context Management**: Efficient handling of conversation history
- **RAG Integration**: Vector database with optimized document retrieval
- **Advanced Capabilities**: Function calling, tool use, and plugin system
- **Memory Management**: Optimized for 512GB unified memory
- **Model Fine-tuning**: Support for DeepSeek model customization

## Architecture

### Memory Management
- **Unified Memory Optimization**:
  - Dynamic batch sizing based on available memory
  - Gradient checkpointing for efficient training
  - Automatic memory cleanup during inference
  - Optimized KV cache management

### Performance Features
- **Inference Optimization**:
  - MPS/Metal acceleration for Apple Silicon
  - Mixed-precision training (FP16/BF16)
  - Efficient attention mechanisms
  - Optimized tensor operations

### Model Training
- **Fine-tuning Capabilities**:
  - LoRA/QLoRA support for efficient adaptation
  - Gradient accumulation for large batch training
  - Custom loss functions and metrics
  - Distributed training support

### RAG System
- **Vector Store**:
  - File-based document storage
  - Efficient similarity search
  - Automatic embedding generation
  - Metadata management

## Project Structure

```
DeepMind/
│
├── backend/                      # FastAPI backend
│   ├── config.py                 # Configuration settings
│   ├── document_processor.py     # Document processing for RAG
│   ├── embedding_service.py      # Embedding generation service
│   ├── main.py                   # Main application entry point
│   ├── main_with_rag.py          # RAG-enhanced application entry
│   ├── model.py                  # Core model implementation
│   ├── model_service.py          # Model serving interface
│   ├── rag_api.py                # RAG API endpoints
│   ├── rag_enhanced_model.py     # RAG-enhanced model implementation
│   ├── rag_model.py              # RAG model core
│   ├── rag_service.py            # RAG service implementation
│   ├── server.py                 # Server implementation
│   ├── vector_store.py           # Vector database implementation
│   ├── requirements.txt          # Python dependencies
│   ├── model/                    # Model management
│   │   ├── optimization.py       # Performance optimizations
│   │   └── training.py           # Fine-tuning logic
│   └── monitoring/               # Performance monitoring tools
│
├── frontend/                     # Next.js frontend
│   ├── components/               # React components
│   ├── monitoring-dashboard/     # Performance monitoring UI
│   ├── pages/                    # Next.js pages
│   ├── public/                   # Static assets
│   ├── src/                      # Source code
│   ├── next.config.js            # Next.js configuration
│   └── tsconfig.json             # TypeScript configuration
│
├── docker/                       # Docker configurations
├── examples/                     # Example implementations
│   └── monitoring_example.py     # Monitoring usage example
│
├── shared/                       # Shared resources (currently empty)
├── .dockerignore                 # Docker ignore file
├── .gitignore                    # Git ignore file
├── Dockerfile                    # Docker configuration
├── docker-compose.yml            # Docker Compose configuration
├── docker-scripts.sh             # Docker helper scripts
└── LICENSE                       # Project license
```

## System Requirements

- **Hardware**:
  - Apple Silicon M-series (optimized for M3 Studio)
  - 512GB unified memory
  - NVMe SSD for fast storage

- **Software**:
  - macOS Sonoma or later
  - Python 3.11+
  - Node.js 18+
  - Docker Desktop for Mac

## Performance Optimization

### Memory Management
1. **Dynamic Batching**:
   ```python
   # Example configuration in config.py
   MAX_BATCH_SIZE = 32
   MIN_BATCH_SIZE = 1
   MEMORY_BUFFER = 0.2  # Keep 20% memory free
   
   def get_optimal_batch_size(available_memory):
       return min(MAX_BATCH_SIZE, max(MIN_BATCH_SIZE, 
              int(available_memory * (1 - MEMORY_BUFFER) / MODEL_MEMORY_FOOTPRINT)))
   ```

2. **Gradient Checkpointing**:
   ```python
   # Example in training.py
   model.gradient_checkpointing_enable()
   model.enable_input_require_grads()
   ```

### Model Fine-tuning

1. **Configuration**:
   ```bash
   # Start fine-tuning
   python train.py \
     --model deepseek-ai/deepseek-coder-33b-instruct \
     --train_batch_size 16 \
     --gradient_accumulation_steps 4 \
     --learning_rate 2e-5 \
     --num_epochs 3 \
     --use_mps
   ```

2. **Monitoring**:
   ```bash
   # Monitor training metrics
   tensorboard --logdir runs/
   ```

## Docker Deployment

1. **Build the Container**:
   ```bash
   ./docker-scripts.sh build
   ```

2. **Start Services**:
   ```bash
   ./docker-scripts.sh start
   ```

3. **Monitor Logs**:
   ```bash
   ./docker-scripts.sh logs
   ```

## Development

### Backend Development

The backend is optimized for high-performance inference and training:
- Token streaming with backpressure handling
- Efficient memory management
- Automatic batch size optimization
- RAG with smart caching

### Model Fine-tuning

1. **Prepare Training Data**:
   ```bash
   python scripts/prepare_data.py \
     --input_file data/raw/training.jsonl \
     --output_dir data/processed
   ```

2. **Start Fine-tuning**:
   ```bash
   python scripts/train.py \
     --model_name deepseek-ai/deepseek-coder-33b-instruct \
     --train_data data/processed/train.pt \
     --eval_data data/processed/eval.pt \
     --output_dir models/fine-tuned
   ```

3. **Monitor Resources**:
   ```bash
   python scripts/monitor.py --log_file training.log
   ```

## Performance Monitoring

- Memory usage tracking
- Inference latency metrics
- Training throughput
- RAG retrieval performance

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
