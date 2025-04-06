# DeepMind Monitoring Dashboard

A real-time monitoring dashboard for the DeepMind project, providing comprehensive visualization of model performance, system metrics, and resource utilization.

## Features

### 1. Real-Time Monitoring
- WebSocket-based real-time updates
- Automatic reconnection handling
- Configurable update intervals

### 2. Performance Metrics
- Model inference time
- Throughput (tokens/second)
- Memory usage
- GPU utilization
- Temperature monitoring

### 3. Model Metrics
- Training loss
- Model perplexity
- Accuracy tracking
- Token count
- Sequence length analysis

### 4. System Resources
- CPU usage
- Memory utilization
- Disk usage
- Network I/O monitoring
- Temperature tracking

### 5. Visualization Types
- **Gauge Charts**: Visual representation of key metrics with color-coded thresholds
- **Radar Charts**: Multi-dimensional view of model performance metrics
- **Heatmaps**: Memory access pattern visualization
- **Line Charts**: Time-series data for tracking metric trends
- **Progress Indicators**: Training progress and resource utilization

## Technology Stack

- **Frontend Framework**: React with TypeScript
- **UI Components**: Chakra UI
- **Visualization**: Chart.js with various plugins
- **State Management**: React Hooks
- **Real-time Updates**: WebSocket
- **Styling**: Emotion (CSS-in-JS)

## Getting Started

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm start
```

3. Open [http://localhost:3000](http://localhost:3000) in your browser

## Project Structure

```
src/
├── components/
│   ├── GaugeChart.tsx    # Circular gauge visualization
│   ├── HeatmapChart.tsx  # Memory pattern heatmap
│   ├── LineChart.tsx     # Time-series line charts
│   ├── ModelPanel.tsx    # Model metrics panel
│   ├── RadarChart.tsx    # Performance radar chart
│   └── SystemPanel.tsx   # System resources panel
├── hooks/
│   ├── useMetrics.ts     # Metrics state management
│   └── useWebSocket.ts   # WebSocket connection handler
└── types/
    └── metrics.ts        # TypeScript type definitions
```

## Configuration

The dashboard can be configured through environment variables:

```env
REACT_APP_WS_URL=ws://localhost:8000/ws  # WebSocket endpoint
REACT_APP_UPDATE_INTERVAL=1000           # Metric update interval (ms)
REACT_APP_RECONNECT_INTERVAL=3000        # WebSocket reconnection interval (ms)
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
