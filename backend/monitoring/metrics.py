#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import psutil
import torch
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from loguru import logger
import json
import os
from pathlib import Path

@dataclass
class PerformanceMetrics:
    """Container for performance-related metrics."""
    inference_time: float  # ms
    tokens_per_second: float
    memory_used: float  # GB
    batch_size: int
    gpu_utilization: Optional[float] = None  # percentage
    temperature: Optional[float] = None  # °C

@dataclass
class ModelMetrics:
    """Container for model-related metrics."""
    perplexity: float
    loss: float
    accuracy: float
    token_count: int
    sequence_length: int

@dataclass
class SystemMetrics:
    """Container for system-related metrics."""
    cpu_usage: float  # percentage
    memory_usage: float  # percentage
    disk_usage: float  # percentage
    network_io: Dict[str, float]  # bytes sent/received
    temperature: Optional[float] = None  # °C

class MetricsCollector:
    """Collects and manages various system and model metrics."""
    
    def __init__(self, log_dir: str = "logs/metrics"):
        """Initialize the metrics collector.
        
        Args:
            log_dir: Directory to store metric logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metric storage
        self.performance_history = []
        self.model_history = []
        self.system_history = []
        
        # Start time for session
        self.start_time = time.time()
        
        logger.info(f"Initialized metrics collector with log directory: {log_dir}")
    
    def collect_performance_metrics(
        self,
        inference_time: float,
        tokens_generated: int,
        batch_size: int
    ) -> PerformanceMetrics:
        """Collect performance-related metrics.
        
        Args:
            inference_time: Time taken for inference (ms)
            tokens_generated: Number of tokens generated
            batch_size: Current batch size
            
        Returns:
            PerformanceMetrics object
        """
        # Calculate tokens per second
        tokens_per_second = tokens_generated / (inference_time / 1000)
        
        # Get memory usage
        memory_used = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
        
        # Get GPU metrics if available
        gpu_util = None
        temperature = None
        if torch.backends.mps.is_available():
            # For Apple Silicon, get system temperature
            try:
                # This is a placeholder - actual implementation would need
                # system-specific temperature monitoring
                temperature = self._get_mac_temperature()
            except:
                pass
        
        metrics = PerformanceMetrics(
            inference_time=inference_time,
            tokens_per_second=tokens_per_second,
            memory_used=memory_used,
            batch_size=batch_size,
            gpu_utilization=gpu_util,
            temperature=temperature
        )
        
        self.performance_history.append(self._metrics_to_dict(metrics))
        return metrics
    
    def collect_model_metrics(
        self,
        loss: float,
        perplexity: float,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> ModelMetrics:
        """Collect model-related metrics.
        
        Args:
            loss: Training/validation loss
            perplexity: Model perplexity
            logits: Model output logits
            targets: Target tokens
            
        Returns:
            ModelMetrics object
        """
        # Calculate accuracy
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == targets).float().mean().item()
        
        metrics = ModelMetrics(
            perplexity=perplexity,
            loss=loss,
            accuracy=accuracy,
            token_count=targets.numel(),
            sequence_length=targets.size(1)
        )
        
        self.model_history.append(self._metrics_to_dict(metrics))
        return metrics
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect system-related metrics.
        
        Returns:
            SystemMetrics object
        """
        # Get CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Get disk usage
        disk = psutil.disk_usage('/')
        disk_usage = disk.percent
        
        # Get network I/O
        net_io = psutil.net_io_counters()
        network_io = {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv
        }
        
        # Get temperature if available
        temperature = None
        if torch.backends.mps.is_available():
            try:
                temperature = self._get_mac_temperature()
            except:
                pass
        
        metrics = SystemMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            network_io=network_io,
            temperature=temperature
        )
        
        self.system_history.append(self._metrics_to_dict(metrics))
        return metrics
    
    def save_metrics(self):
        """Save collected metrics to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save performance metrics
        if self.performance_history:
            self._save_metric_file(
                "performance",
                self.performance_history,
                timestamp
            )
        
        # Save model metrics
        if self.model_history:
            self._save_metric_file(
                "model",
                self.model_history,
                timestamp
            )
        
        # Save system metrics
        if self.system_history:
            self._save_metric_file(
                "system",
                self.system_history,
                timestamp
            )
        
        logger.info(f"Saved metrics to {self.log_dir}")
    
    def _save_metric_file(self, name: str, data: list, timestamp: str):
        """Save metrics to a JSON file.
        
        Args:
            name: Metric type name
            data: List of metric dictionaries
            timestamp: Timestamp string
        """
        filename = self.log_dir / f"{name}_metrics_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _metrics_to_dict(self, metrics: Any) -> Dict[str, Any]:
        """Convert metrics object to dictionary with timestamp.
        
        Args:
            metrics: Metrics object
            
        Returns:
            Dictionary with metrics and timestamp
        """
        return {
            'timestamp': datetime.now().isoformat(),
            **metrics.__dict__
        }
    
    def _get_mac_temperature(self) -> Optional[float]:
        """Get system temperature for Mac.
        
        Returns:
            Temperature in Celsius if available
        """
        # This is a placeholder - actual implementation would need
        # system-specific temperature monitoring
        # For Mac Studio, you might use a third-party tool like
        # smckit or osx-cpu-temp
        return None
