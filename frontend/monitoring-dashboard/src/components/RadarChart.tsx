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
  RadarController,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Legend,
  Tooltip
} from 'chart.js';

// Register Chart.js components
Chart.register(
  RadarController,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Legend,
  Tooltip
);

interface RadarMetric {
  label: string;
  value: number;
  maxValue: number;
}

interface RadarChartProps {
  title: string;
  metrics: RadarMetric[];
  showLegend?: boolean;
}

export const RadarChart = ({
  title,
  metrics,
  showLegend = false
}: RadarChartProps) => {
  const chartRef = useRef<HTMLCanvasElement>(null);
  const chartInstance = useRef<Chart | null>(null);
  
  // Get theme colors
  const [blue500, blue200] = useToken('colors', ['blue.500', 'blue.200']);
  const textColor = useColorModeValue('gray.800', 'white');
  const gridColor = useColorModeValue('gray.200', 'gray.600');
  
  useEffect(() => {
    if (!chartRef.current) return;
    
    // Destroy previous chart instance
    if (chartInstance.current) {
      chartInstance.current.destroy();
    }
    
    // Normalize values to percentages
    const normalizedValues = metrics.map(m => (m.value / m.maxValue) * 100);
    
    // Chart configuration
    const config: ChartConfiguration = {
      type: 'radar',
      data: {
        labels: metrics.map(m => m.label),
        datasets: [{
          label: 'Current Performance',
          data: normalizedValues,
          backgroundColor: `${blue500}33`,
          borderColor: blue500,
          borderWidth: 2,
          pointBackgroundColor: blue500,
          pointBorderColor: blue500,
          pointHoverBackgroundColor: blue200,
          pointHoverBorderColor: blue500
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          r: {
            beginAtZero: true,
            max: 100,
            ticks: {
              stepSize: 20,
              color: textColor
            },
            grid: {
              color: gridColor
            },
            pointLabels: {
              color: textColor,
              font: {
                size: 12
              }
            }
          }
        },
        plugins: {
          legend: {
            display: showLegend,
            position: 'top',
            labels: {
              color: textColor
            }
          },
          tooltip: {
            callbacks: {
              label: (context: any) => {
                const index = context.dataIndex;
                const metric = metrics[index];
                return `${metric.label}: ${metric.value.toFixed(1)}/${metric.maxValue}`;
              }
            }
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
  }, [metrics, showLegend, textColor, gridColor, blue500, blue200]);
  
  return (
    <Box>
      <Text fontSize="sm" mb={2}>{title}</Text>
      <Box h="300px">
        <canvas ref={chartRef} />
      </Box>
    </Box>
  );
};
