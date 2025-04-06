#!/usr/bin/env python
# -*- coding: utf-8 -*-

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import time
from loguru import logger

from .service import MonitoringService

class DashboardAPI:
    """API for the real-time monitoring dashboard."""
    
    def __init__(self):
        """Initialize the dashboard API."""
        self.app = FastAPI(title="DeepMind Monitoring Dashboard")
        self.monitor = MonitoringService()
        self.active_connections: List[WebSocket] = []
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # In production, replace with actual origins
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Register routes
        self._setup_routes()
        
        # Start monitoring service
        self.monitor.start()
        
        # Start metrics broadcast
        asyncio.create_task(self._broadcast_metrics())
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/")
        async def root():
            return {"status": "ok", "service": "DeepMind Monitoring Dashboard"}
        
        @self.app.get("/metrics/latest")
        async def get_latest_metrics():
            """Get the latest metrics."""
            return self.monitor.get_latest_metrics()
        
        @self.app.get("/metrics/history")
        async def get_metrics_history(
            metric_type: str,
            limit: Optional[int] = 100
        ):
            """Get historical metrics.
            
            Args:
                metric_type: Type of metrics (performance/model/system)
                limit: Maximum number of records to return
            """
            if metric_type == "performance":
                data = self.monitor.visualizer.performance_data
            elif metric_type == "model":
                data = self.monitor.visualizer.model_data
            elif metric_type == "system":
                data = self.monitor.visualizer.system_data
            else:
                return {"error": "Invalid metric type"}
            
            return {"data": data[-limit:] if limit else data}
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time metrics."""
            await websocket.accept()
            self.active_connections.append(websocket)
            try:
                while True:
                    # Keep connection alive
                    await websocket.receive_text()
            except:
                self.active_connections.remove(websocket)
    
    async def _broadcast_metrics(self):
        """Broadcast metrics to all connected clients."""
        while True:
            try:
                # Get latest metrics
                metrics = self.monitor.get_latest_metrics()
                metrics["timestamp"] = datetime.now().isoformat()
                
                # Broadcast to all connections
                for connection in self.active_connections:
                    try:
                        await connection.send_json(metrics)
                    except:
                        # Remove dead connections
                        self.active_connections.remove(connection)
                
                # Wait before next broadcast
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error broadcasting metrics: {e}")
                await asyncio.sleep(1)
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application.
        
        Returns:
            FastAPI application
        """
        return self.app
