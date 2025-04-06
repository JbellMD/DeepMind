import React, { useEffect, useRef } from 'react';
import {
  Box,
  Text,
  useColorModeValue,
  useToken
} from '@chakra-ui/react';
import {
  Chart,
  ChartConfiguration,
  ScatterController,
  ScatterElement,
  LinearScale,
  Tooltip
} from 'chart.js';
import 'chartjs-adapter-date-fns';

// Register Chart.js components
Chart.register(
  ScatterController,
  ScatterElement,
  LinearScale,
  Tooltip
);

interface HeatmapPoint {
  x: number;
  y: number;
  value: number;
}

interface HeatmapChartProps {
  title: string;
  data: HeatmapPoint[];
  xLabel: string;
  yLabel: string;
  maxValue?: number;
}

export const HeatmapChart = ({
  title,
  data,
  xLabel,
  yLabel,
  maxValue = 100
}: HeatmapChartProps) => {
  const chartRef = useRef<HTMLCanvasElement>(null);
  const chartInstance = useRef<Chart | null>(null);
  
  // Get theme colors
  const [red500, yellow500, green500] = useToken('colors', ['red.500', 'yellow.500', 'green.500']);
  const textColor = useColorModeValue('gray.800', 'white');
  const gridColor = useColorModeValue('gray.200', 'gray.600');
  
  const getColor = (value: number) => {
    const normalizedValue = value / maxValue;
    if (normalizedValue < 0.5) {
      return green500;
    } else if (normalizedValue < 0.8) {
      return yellow500;
    }
    return red500;
  };
  
  useEffect(() => {
    if (!chartRef.current) return;
    
    // Destroy previous chart instance
    if (chartInstance.current) {
      chartInstance.current.destroy();
    }
    
    // Prepare data points with colors
    const points = data.map(point => ({
      x: point.x,
      y: point.y,
      value: point.value,
      pointBackgroundColor: getColor(point.value),
      pointRadius: 10
    }));
    
    // Chart configuration
    const config: ChartConfiguration = {
      type: 'scatter',
      data: {
        datasets: [{
          data: points,
          backgroundColor: points.map(p => p.pointBackgroundColor),
          pointRadius: points.map(p => p.pointRadius),
          pointHoverRadius: 12
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
          duration: 0
        },
        scales: {
          x: {
            title: {
              display: true,
              text: xLabel,
              color: textColor
            },
            grid: {
              color: gridColor
            },
            ticks: {
              color: textColor
            }
          },
          y: {
            title: {
              display: true,
              text: yLabel,
              color: textColor
            },
            grid: {
              color: gridColor
            },
            ticks: {
              color: textColor
            }
          }
        },
        plugins: {
          tooltip: {
            callbacks: {
              label: (context: any) => {
                const point = data[context.dataIndex];
                return `${xLabel}: ${point.x}, ${yLabel}: ${point.y}, Value: ${point.value}%`;
              }
            }
          },
          legend: {
            display: false
          }
        }
      }
    };
    
    // Create new chart
    chartInstance.current = new Chart(chartRef.current, config);
    
    // Cleanup
    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }
    };
  }, [data, title, xLabel, yLabel, maxValue, textColor, gridColor, red500, yellow500, green500]);
  
  return (
    <Box>
      <Text fontSize="sm" mb={2}>{title}</Text>
      <Box h="300px">
        <canvas ref={chartRef} />
      </Box>
    </Box>
  );
};
