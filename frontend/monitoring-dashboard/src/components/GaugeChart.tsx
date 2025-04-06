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
  DoughnutController,
  ArcElement,
  Tooltip
} from 'chart.js';

// Register Chart.js components
Chart.register(
  DoughnutController,
  ArcElement,
  Tooltip
);

interface GaugeChartProps {
  title: string;
  value: number;
  maxValue?: number;
  warningThreshold?: number;
  criticalThreshold?: number;
  unit?: string;
}

export const GaugeChart = ({
  title,
  value,
  maxValue = 100,
  warningThreshold = 70,
  criticalThreshold = 90,
  unit = '%'
}: GaugeChartProps) => {
  const chartRef = useRef<HTMLCanvasElement>(null);
  const chartInstance = useRef<Chart | null>(null);
  
  // Get theme colors
  const [green500, yellow500, red500] = useToken(
    'colors',
    ['green.500', 'yellow.500', 'red.500']
  );
  const backgroundColor = useColorModeValue('gray.100', 'gray.700');
  const textColor = useColorModeValue('gray.800', 'white');
  
  const getColor = (value: number) => {
    const percentage = (value / maxValue) * 100;
    if (percentage >= criticalThreshold) return red500;
    if (percentage >= warningThreshold) return yellow500;
    return green500;
  };
  
  useEffect(() => {
    if (!chartRef.current) return;
    
    // Destroy previous chart instance
    if (chartInstance.current) {
      chartInstance.current.destroy();
    }
    
    // Normalize value
    const normalizedValue = Math.min(Math.max(0, value), maxValue);
    const percentage = (normalizedValue / maxValue) * 100;
    const color = getColor(normalizedValue);
    
    // Chart configuration
    const config: ChartConfiguration = {
      type: 'doughnut',
      data: {
        datasets: [{
          data: [percentage, 100 - percentage],
          backgroundColor: [color, backgroundColor],
          borderWidth: 0,
          circumference: 180,
          rotation: 270
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '75%',
        plugins: {
          tooltip: {
            enabled: false
          },
          legend: {
            display: false
          }
        },
        layout: {
          padding: {
            top: 10
          }
        }
      },
      plugins: [{
        id: 'gaugeText',
        afterDraw: (chart: any) => {
          const { ctx, chartArea: { top, bottom, left, right, width, height } } = chart;
          
          ctx.save();
          
          // Draw value
          const text = `${normalizedValue.toFixed(1)}${unit}`;
          ctx.font = 'bold 24px sans-serif';
          ctx.fillStyle = textColor;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(text, width / 2, height * 0.6);
          
          // Draw thresholds
          ctx.font = '12px sans-serif';
          ctx.fillStyle = textColor;
          
          // Min value (0)
          ctx.save();
          ctx.translate(width * 0.15, height * 0.65);
          ctx.rotate(-Math.PI / 4);
          ctx.fillText('0' + unit, 0, 0);
          ctx.restore();
          
          // Max value
          ctx.save();
          ctx.translate(width * 0.85, height * 0.65);
          ctx.rotate(Math.PI / 4);
          ctx.fillText(maxValue + unit, 0, 0);
          ctx.restore();
          
          ctx.restore();
        }
      }]
    };
    
    // Create new chart
    chartInstance.current = new Chart(chartRef.current, config);
    
    // Cleanup
    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }
    };
  }, [value, maxValue, warningThreshold, criticalThreshold, unit, backgroundColor, textColor, green500, yellow500, red500]);
  
  return (
    <Box>
      <Text fontSize="sm" mb={2}>{title}</Text>
      <Box h="150px">
        <canvas ref={chartRef} />
      </Box>
    </Box>
  );
};
