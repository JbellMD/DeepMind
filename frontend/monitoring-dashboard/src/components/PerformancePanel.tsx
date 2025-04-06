import React from 'react';
import {
  VStack,
  Heading,
  SimpleGrid,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  useColorModeValue
} from '@chakra-ui/react';
import { LineChart } from './LineChart';

interface PerformanceMetrics {
  inference_time: number;
  tokens_per_second: number;
  memory_used: number;
  batch_size: number;
  gpu_utilization?: number;
  temperature?: number;
}

export const PerformancePanel = ({ metrics }: { metrics?: PerformanceMetrics }) => {
  const statBg = useColorModeValue('gray.50', 'gray.600');
  
  return (
    <VStack spacing={6} align="stretch">
      <Heading size="md">Performance Metrics</Heading>
      
      <SimpleGrid columns={{ base: 2, md: 3 }} spacing={4}>
        {/* Inference Time */}
        <Stat bg={statBg} p={4} rounded="md">
          <StatLabel>Inference Time</StatLabel>
          <StatNumber>{metrics?.inference_time?.toFixed(2) || '-'}</StatNumber>
          <StatHelpText>milliseconds</StatHelpText>
        </Stat>
        
        {/* Tokens per Second */}
        <Stat bg={statBg} p={4} rounded="md">
          <StatLabel>Throughput</StatLabel>
          <StatNumber>{metrics?.tokens_per_second?.toFixed(1) || '-'}</StatNumber>
          <StatHelpText>tokens/second</StatHelpText>
        </Stat>
        
        {/* Memory Usage */}
        <Stat bg={statBg} p={4} rounded="md">
          <StatLabel>Memory Usage</StatLabel>
          <StatNumber>{metrics?.memory_used?.toFixed(1) || '-'}</StatNumber>
          <StatHelpText>GB</StatHelpText>
        </Stat>
        
        {/* Batch Size */}
        <Stat bg={statBg} p={4} rounded="md">
          <StatLabel>Batch Size</StatLabel>
          <StatNumber>{metrics?.batch_size || '-'}</StatNumber>
          <StatHelpText>samples</StatHelpText>
        </Stat>
        
        {/* GPU Utilization */}
        {metrics?.gpu_utilization !== undefined && (
          <Stat bg={statBg} p={4} rounded="md">
            <StatLabel>GPU Usage</StatLabel>
            <StatNumber>{metrics.gpu_utilization.toFixed(1)}%</StatNumber>
            <StatHelpText>utilization</StatHelpText>
          </Stat>
        )}
        
        {/* Temperature */}
        {metrics?.temperature !== undefined && (
          <Stat bg={statBg} p={4} rounded="md">
            <StatLabel>Temperature</StatLabel>
            <StatNumber>{metrics.temperature.toFixed(1)}°C</StatNumber>
            <StatHelpText>system</StatHelpText>
          </Stat>
        )}
      </SimpleGrid>
      
      {/* Performance Charts */}
      <SimpleGrid columns={{ base: 1, xl: 2 }} spacing={4}>
        <LineChart
          title="Inference Time"
          data={[{ x: new Date(), y: metrics?.inference_time || 0 }]}
          yLabel="ms"
        />
        <LineChart
          title="Throughput"
          data={[{ x: new Date(), y: metrics?.tokens_per_second || 0 }]}
          yLabel="tokens/s"
        />
      </SimpleGrid>
    </VStack>
  );
};
