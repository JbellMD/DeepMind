#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json
from loguru import logger

from .metrics import MetricsCollector
from .visualizer import MetricsVisualizer

class MonitoringService:
    """Service for collecting and visualizing system metrics."""
    
    def __init__(
        self,
        log_dir: str = "logs/metrics",
        report_dir: str = "reports",
        collection_interval: int = 60  # seconds
    ):
        """Initialize the monitoring service.
        
        Args:
            log_dir: Directory to store metric logs
            report_dir: Directory to store reports
            collection_interval: Interval between metric collections
        """
        self.log_dir = Path(log_dir)
        self.report_dir = Path(report_dir)
        self.collection_interval = collection_interval
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.metrics_collector = MetricsCollector(str(self.log_dir))
        self.visualizer = MetricsVisualizer(str(self.log_dir))
        
        # Threading control
        self._stop_event = threading.Event()
        self._collection_thread = None
        
        logger.info(
            f"Initialized monitoring service with:\n"
            f"  Log directory: {self.log_dir}\n"
            f"  Report directory: {self.report_dir}\n"
            f"  Collection interval: {collection_interval}s"
        )
    
    def start(self):
        """Start the monitoring service."""
        if self._collection_thread is not None and self._collection_thread.is_alive():
            logger.warning("Monitoring service is already running")
            return
        
        self._stop_event.clear()
        self._collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True
        )
        self._collection_thread.start()
        
        logger.info("Started monitoring service")
    
    def stop(self):
        """Stop the monitoring service."""
        if self._collection_thread is None or not self._collection_thread.is_alive():
            logger.warning("Monitoring service is not running")
            return
        
        self._stop_event.set()
        self._collection_thread.join()
        self._collection_thread = None
        
        # Save final metrics
        self.metrics_collector.save_metrics()
        
        logger.info("Stopped monitoring service")
    
    def _collection_loop(self):
        """Main collection loop."""
        while not self._stop_event.is_set():
            try:
                # Collect system metrics
                self.metrics_collector.collect_system_metrics()
                
                # Save metrics periodically
                self.metrics_collector.save_metrics()
                
                # Wait for next collection
                self._stop_event.wait(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
    
    def record_inference(
        self,
        inference_time: float,
        tokens_generated: int,
        batch_size: int
    ):
        """Record inference-related metrics.
        
        Args:
            inference_time: Time taken for inference (ms)
            tokens_generated: Number of tokens generated
            batch_size: Current batch size
        """
        self.metrics_collector.collect_performance_metrics(
            inference_time=inference_time,
            tokens_generated=tokens_generated,
            batch_size=batch_size
        )
    
    def record_training(
        self,
        loss: float,
        perplexity: float,
        logits: Any,
        targets: Any
    ):
        """Record training-related metrics.
        
        Args:
            loss: Training loss
            perplexity: Model perplexity
            logits: Model output logits
            targets: Target tokens
        """
        self.metrics_collector.collect_model_metrics(
            loss=loss,
            perplexity=perplexity,
            logits=logits,
            targets=targets
        )
    
    def generate_report(
        self,
        report_name: Optional[str] = None
    ):
        """Generate a monitoring report.
        
        Args:
            report_name: Optional name for the report
        """
        if report_name is None:
            report_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report_dir = self.report_dir / report_name
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate visualizations
        self.visualizer.generate_report(str(report_dir))
        
        # Create report summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'metrics_collected': {
                'performance': len(self.visualizer.performance_data),
                'model': len(self.visualizer.model_data),
                'system': len(self.visualizer.system_data)
            },
            'collection_interval': self.collection_interval,
            'report_location': str(report_dir)
        }
        
        with open(report_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Generated monitoring report: {report_dir}")
        return str(report_dir)
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get the most recent metrics.
        
        Returns:
            Dictionary containing latest metrics
        """
        latest = {}
        
        # Get latest performance metrics
        if self.visualizer.performance_data:
            latest['performance'] = self.visualizer.performance_data[-1]
        
        # Get latest model metrics
        if self.visualizer.model_data:
            latest['model'] = self.visualizer.model_data[-1]
        
        # Get latest system metrics
        if self.visualizer.system_data:
            latest['system'] = self.visualizer.system_data[-1]
        
        return latest
