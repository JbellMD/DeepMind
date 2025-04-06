import { useState, useCallback } from 'react';

interface PerformanceMetrics {
  inference_time: number;
  tokens_per_second: number;
  memory_used: number;
  batch_size: number;
  gpu_utilization?: number;
  temperature?: number;
}

interface ModelMetrics {
  perplexity: number;
  loss: number;
  accuracy: number;
  token_count: number;
  sequence_length: number;
}

interface SystemMetrics {
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  network_io: {
    bytes_sent: number;
    bytes_recv: number;
  };
  temperature?: number;
}

interface Metrics {
  performance?: PerformanceMetrics;
  model?: ModelMetrics;
  system?: SystemMetrics;
  timestamp?: string;
}

export const useMetrics = () => {
  const [metrics, setMetrics] = useState<Metrics>({});
  
  const updateMetrics = useCallback((newMetrics: Metrics) => {
    setMetrics(current => ({
      ...current,
      ...newMetrics,
      // Keep history of last 100 data points for each metric type
      performance: {
        ...(current.performance || {}),
        ...(newMetrics.performance || {})
      },
      model: {
        ...(current.model || {}),
        ...(newMetrics.model || {})
      },
      system: {
        ...(current.system || {}),
        ...(newMetrics.system || {})
      }
    }));
  }, []);
  
  return {
    metrics,
    updateMetrics
  };
};
