import React from 'react';
import {
  VStack,
  Heading,
  SimpleGrid,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  Progress,
  useColorModeValue,
  Box
} from '@chakra-ui/react';
import { LineChart } from './LineChart';
import { GaugeChart } from './GaugeChart';
import { RadarChart } from './RadarChart';
import { HeatmapChart } from './HeatmapChart';

interface ModelMetrics {
  perplexity: number;
  loss: number;
  accuracy: number;
  token_count: number;
  sequence_length: number;
  memory_pattern?: Array<{ x: number; y: number; value: number }>;
}

export const ModelPanel = ({ metrics }: { metrics?: ModelMetrics }) => {
  const statBg = useColorModeValue('gray.50', 'gray.600');
  
  // Prepare radar metrics
  const radarMetrics = metrics ? [
    { label: 'Accuracy', value: metrics.accuracy * 100, maxValue: 100 },
    { label: 'Perplexity', value: Math.min(metrics.perplexity, 10), maxValue: 10 },
    { label: 'Loss', value: Math.min(metrics.loss, 5), maxValue: 5 },
    { label: 'Sequence Coverage', value: (metrics.sequence_length / 2048) * 100, maxValue: 100 },
    { label: 'Token Efficiency', value: Math.min((metrics.token_count / 1000), 100), maxValue: 100 }
  ] : [];
  
  return (
    <VStack spacing={6} align="stretch">
      <Heading size="md">Model Metrics</Heading>
      
      {/* Key Metrics with Gauges */}
      <SimpleGrid columns={{ base: 2, lg: 3 }} spacing={4}>
        <GaugeChart
          title="Model Accuracy"
          value={metrics?.accuracy ? metrics.accuracy * 100 : 0}
          warningThreshold={60}
          criticalThreshold={40}
        />
        
        <GaugeChart
          title="Perplexity"
          value={metrics?.perplexity || 0}
          maxValue={10}
          warningThreshold={60}
          criticalThreshold={80}
          unit=""
        />
        
        <GaugeChart
          title="Loss"
          value={metrics?.loss || 0}
          maxValue={5}
          warningThreshold={60}
          criticalThreshold={80}
          unit=""
        />
      </SimpleGrid>
      
      {/* Performance Radar */}
      <Box
        bg={useColorModeValue('white', 'gray.700')}
        p={4}
        rounded="lg"
        shadow="base"
      >
        <RadarChart
          title="Model Performance Overview"
          metrics={radarMetrics}
        />
      </Box>
      
      {/* Memory Pattern Heatmap */}
      {metrics?.memory_pattern && (
        <Box
          bg={useColorModeValue('white', 'gray.700')}
          p={4}
          rounded="lg"
          shadow="base"
        >
          <HeatmapChart
            title="Memory Access Pattern"
            data={metrics.memory_pattern}
            xLabel="Time"
            yLabel="Memory Block"
          />
        </Box>
      )}
      
      {/* Training Progress */}
      <SimpleGrid columns={{ base: 1, xl: 2 }} spacing={4}>
        <LineChart
          title="Training Loss"
          data={[{ x: new Date(), y: metrics?.loss || 0 }]}
          yLabel="loss"
        />
        <LineChart
          title="Model Accuracy"
          data={[{ x: new Date(), y: (metrics?.accuracy || 0) * 100 }]}
          yLabel="%"
        />
      </SimpleGrid>
      
      {/* Detailed Stats */}
      <SimpleGrid columns={{ base: 2, md: 3 }} spacing={4}>
        <Stat bg={statBg} p={4} rounded="md">
          <StatLabel>Token Count</StatLabel>
          <StatNumber>{metrics?.token_count?.toLocaleString() || '-'}</StatNumber>
          <StatHelpText>total tokens</StatHelpText>
        </Stat>
        
        <Stat bg={statBg} p={4} rounded="md">
          <StatLabel>Sequence Length</StatLabel>
          <StatNumber>{metrics?.sequence_length || '-'}</StatNumber>
          <StatHelpText>tokens/sequence</StatHelpText>
        </Stat>
        
        <Stat bg={statBg} p={4} rounded="md">
          <StatLabel>Training Progress</StatLabel>
          <StatNumber>
            {metrics ? Math.min(((metrics.token_count || 0) / 1000000) * 100, 100).toFixed(1) : '-'}%
          </StatNumber>
          <StatHelpText>
            <Progress
              value={metrics ? Math.min(((metrics.token_count || 0) / 1000000) * 100, 100) : 0}
              size="xs"
              colorScheme="green"
            />
          </StatHelpText>
        </Stat>
      </SimpleGrid>
    </VStack>
  );
};
