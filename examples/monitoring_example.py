#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.monitoring.service import MonitoringService
import time
import torch
from loguru import logger

def simulate_training():
    """Simulate a training process with metrics."""
    # Create random training data
    batch_size = 16
    seq_length = 512
    vocab_size = 32000
    
    logits = torch.randn(batch_size, seq_length, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Simulate metrics
    loss = float(torch.rand(1))
    perplexity = float(torch.exp(torch.tensor(loss)))
    
    return loss, perplexity, logits, targets

def simulate_inference():
    """Simulate an inference process with metrics."""
    # Simulate inference time (100-500ms)
    inference_time = float(torch.randint(100, 500, (1,)))
    
    # Simulate generated tokens (10-50 tokens)
    tokens_generated = int(torch.randint(10, 50, (1,)))
    
    # Simulate batch size (1-16)
    batch_size = int(torch.randint(1, 16, (1,)))
    
    return inference_time, tokens_generated, batch_size

def main():
    # Initialize monitoring service
    monitor = MonitoringService(
        log_dir="logs/metrics",
        report_dir="reports",
        collection_interval=5  # 5 seconds for demo
    )
    
    try:
        # Start monitoring
        logger.info("Starting monitoring service...")
        monitor.start()
        
        # Simulate some workload
        logger.info("Simulating workload...")
        for i in range(10):
            # Simulate training
            loss, perplexity, logits, targets = simulate_training()
            monitor.record_training(loss, perplexity, logits, targets)
            logger.info(f"Training step {i+1}: loss={loss:.4f}, perplexity={perplexity:.4f}")
            
            # Simulate inference
            inference_time, tokens_generated, batch_size = simulate_inference()
            monitor.record_inference(inference_time, tokens_generated, batch_size)
            logger.info(
                f"Inference step {i+1}: "
                f"time={inference_time:.2f}ms, "
                f"tokens={tokens_generated}, "
                f"batch_size={batch_size}"
            )
            
            # Wait a bit
            time.sleep(2)
        
        # Generate report
        logger.info("Generating monitoring report...")
        report_path = monitor.generate_report("example_run")
        logger.info(f"Report generated at: {report_path}")
        
        # Get latest metrics
        latest = monitor.get_latest_metrics()
        logger.info("Latest metrics:")
        logger.info(f"Performance: {latest.get('performance', 'N/A')}")
        logger.info(f"Model: {latest.get('model', 'N/A')}")
        logger.info(f"System: {latest.get('system', 'N/A')}")
    
    finally:
        # Stop monitoring
        logger.info("Stopping monitoring service...")
        monitor.stop()

if __name__ == "__main__":
    main()
