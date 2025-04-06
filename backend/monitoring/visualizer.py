#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
from loguru import logger

class MetricsVisualizer:
    """Visualizes collected metrics using interactive plots."""
    
    def __init__(self, metrics_dir: str = "logs/metrics"):
        """Initialize the visualizer.
        
        Args:
            metrics_dir: Directory containing metric logs
        """
        self.metrics_dir = Path(metrics_dir)
        self.performance_data = []
        self.model_data = []
        self.system_data = []
        
        # Load available metrics
        self._load_metrics()
    
    def _load_metrics(self):
        """Load metrics from JSON files."""
        # Load performance metrics
        for file in self.metrics_dir.glob("performance_metrics_*.json"):
            with open(file) as f:
                self.performance_data.extend(json.load(f))
        
        # Load model metrics
        for file in self.metrics_dir.glob("model_metrics_*.json"):
            with open(file) as f:
                self.model_data.extend(json.load(f))
        
        # Load system metrics
        for file in self.metrics_dir.glob("system_metrics_*.json"):
            with open(file) as f:
                self.system_data.extend(json.load(f))
        
        logger.info(f"Loaded metrics from {self.metrics_dir}")
    
    def create_performance_dashboard(
        self,
        output_file: Optional[str] = None
    ) -> go.Figure:
        """Create an interactive performance dashboard.
        
        Args:
            output_file: Optional file to save the dashboard HTML
            
        Returns:
            Plotly figure object
        """
        if not self.performance_data:
            logger.warning("No performance data available")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(self.performance_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Inference Time',
                'Tokens per Second',
                'Memory Usage',
                'Batch Size',
                'GPU Utilization',
                'Temperature'
            )
        )
        
        # Add traces
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['inference_time'],
                      name='Inference Time (ms)'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['tokens_per_second'],
                      name='Tokens/s'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['memory_used'],
                      name='Memory (GB)'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['batch_size'],
                      name='Batch Size'),
            row=2, col=2
        )
        
        if 'gpu_utilization' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['gpu_utilization'],
                          name='GPU %'),
                row=3, col=1
            )
        
        if 'temperature' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['temperature'],
                          name='Temp (°C)'),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1000,
            title_text="Performance Metrics Dashboard",
            showlegend=True
        )
        
        if output_file:
            fig.write_html(output_file)
            logger.info(f"Saved performance dashboard to {output_file}")
        
        return fig
    
    def create_model_dashboard(
        self,
        output_file: Optional[str] = None
    ) -> go.Figure:
        """Create an interactive model metrics dashboard.
        
        Args:
            output_file: Optional file to save the dashboard HTML
            
        Returns:
            Plotly figure object
        """
        if not self.model_data:
            logger.warning("No model data available")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(self.model_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Loss',
                'Perplexity',
                'Accuracy',
                'Token Count'
            )
        )
        
        # Add traces
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['loss'],
                      name='Loss'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['perplexity'],
                      name='Perplexity'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['accuracy'],
                      name='Accuracy'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['token_count'],
                      name='Tokens'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Model Metrics Dashboard",
            showlegend=True
        )
        
        if output_file:
            fig.write_html(output_file)
            logger.info(f"Saved model dashboard to {output_file}")
        
        return fig
    
    def create_system_dashboard(
        self,
        output_file: Optional[str] = None
    ) -> go.Figure:
        """Create an interactive system metrics dashboard.
        
        Args:
            output_file: Optional file to save the dashboard HTML
            
        Returns:
            Plotly figure object
        """
        if not self.system_data:
            logger.warning("No system data available")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(self.system_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'CPU Usage',
                'Memory Usage',
                'Disk Usage',
                'Network I/O'
            )
        )
        
        # Add traces
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['cpu_usage'],
                      name='CPU %'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['memory_usage'],
                      name='Memory %'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['disk_usage'],
                      name='Disk %'),
            row=2, col=1
        )
        
        # Network I/O
        network_df = pd.DataFrame([d['network_io'] for d in df['network_io']])
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=network_df['bytes_sent']/1e6,
                      name='MB Sent'),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=network_df['bytes_recv']/1e6,
                      name='MB Received'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="System Metrics Dashboard",
            showlegend=True
        )
        
        if output_file:
            fig.write_html(output_file)
            logger.info(f"Saved system dashboard to {output_file}")
        
        return fig
    
    def generate_report(
        self,
        output_dir: str = "reports"
    ):
        """Generate a comprehensive HTML report with all metrics.
        
        Args:
            output_dir: Directory to save the report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate individual dashboards
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.create_performance_dashboard(
            output_dir / f"performance_dashboard_{timestamp}.html"
        )
        
        self.create_model_dashboard(
            output_dir / f"model_dashboard_{timestamp}.html"
        )
        
        self.create_system_dashboard(
            output_dir / f"system_dashboard_{timestamp}.html"
        )
        
        logger.info(f"Generated reports in {output_dir}")
