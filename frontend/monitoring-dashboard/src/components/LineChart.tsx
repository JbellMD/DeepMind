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
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  TimeScale,
  Legend,
  Tooltip
} from 'chart.js';
import 'chartjs-adapter-date-fns';

// Register Chart.js components
Chart.register(
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  TimeScale,
  Legend,
  Tooltip
);

interface DataPoint {
  x: Date;
  y: number;
  name?: string;
}

interface LineChartProps {
  title: string;
  data: DataPoint[];
  yLabel: string;
  showLegend?: boolean;
}

export const LineChart = ({
  title,
  data,
  yLabel,
  showLegend = false
}: LineChartProps) => {
  const chartRef = useRef<HTMLCanvasElement>(null);
  const chartInstance = useRef<Chart | null>(null);
  
  // Get theme colors
  const [blue500] = useToken('colors', ['blue.500']);
  const textColor = useColorModeValue('gray.800', 'white');
  const gridColor = useColorModeValue('gray.200', 'gray.600');
  
  useEffect(() => {
    if (!chartRef.current) return;
    
    // Destroy previous chart instance
    if (chartInstance.current) {
      chartInstance.current.destroy();
    }
    
    // Group data by series name
    const series = data.reduce((acc, point) => {
      const name = point.name || 'default';
      if (!acc[name]) {
        acc[name] = [];
      }
      acc[name].push(point);
      return acc;
    }, {} as Record<string, DataPoint[]>);
    
    // Create datasets
    const datasets = Object.entries(series).map(([name, points], index) => ({
      label: name === 'default' ? title : name,
      data: points.map(p => ({ x: p.x, y: p.y })),
      borderColor: index === 0 ? blue500 : `hsl(${index * 60}, 70%, 50%)`,
      backgroundColor: index === 0 ? blue500 : `hsl(${index * 60}, 70%, 50%)`,
      tension: 0.4,
      pointRadius: 0,
      borderWidth: 2
    }));
    
    // Chart configuration
    const config: ChartConfiguration = {
      type: 'line',
      data: {
        datasets
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
          duration: 0
        },
        scales: {
          x: {
            type: 'time',
            time: {
              unit: 'second'
            },
            grid: {
              color: gridColor
            },
            ticks: {
              color: textColor
            }
          },
          y: {
            beginAtZero: true,
            grid: {
              color: gridColor
            },
            ticks: {
              color: textColor
            },
            title: {
              display: true,
              text: yLabel,
              color: textColor
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
            mode: 'index',
            intersect: false
          }
        },
        interaction: {
          mode: 'nearest',
          axis: 'x',
          intersect: false
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
  }, [data, title, yLabel, showLegend, blue500, textColor, gridColor]);
  
  return (
    <Box>
      <Text fontSize="sm" mb={2}>{title}</Text>
      <Box h="200px">
        <canvas ref={chartRef} />
      </Box>
    </Box>
  );
};
