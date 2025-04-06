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
  useColorModeValue
} from '@chakra-ui/react';
import { LineChart } from './LineChart';

interface ModelMetrics {
  perplexity: number;
  loss: number;
  accuracy: number;
  token_count: number;
  sequence_length: number;
}

export const ModelPanel = ({ metrics }: { metrics?: ModelMetrics }) => {
  const statBg = useColorModeValue('gray.50', 'gray.600');
  
  return (
    <VStack spacing={6} align="stretch">
      <Heading size="md">Model Metrics</Heading>
      
      <SimpleGrid columns={{ base: 2, md: 3 }} spacing={4}>
        {/* Loss */}
        <Stat bg={statBg} p={4} rounded="md">
          <StatLabel>Loss</StatLabel>
          <StatNumber>{metrics?.loss?.toFixed(4) || '-'}</StatNumber>
          <StatHelpText>
            <Progress
              value={Math.max(0, 100 - (metrics?.loss || 0) * 100)}
              size="xs"
              colorScheme="green"
            />
          </StatHelpText>
        </Stat>
        
        {/* Perplexity */}
        <Stat bg={statBg} p={4} rounded="md">
          <StatLabel>Perplexity</StatLabel>
          <StatNumber>{metrics?.perplexity?.toFixed(2) || '-'}</StatNumber>
          <StatHelpText>
            <Progress
              value={Math.max(0, 100 - (metrics?.perplexity || 0) * 10)}
              size="xs"
              colorScheme="blue"
            />
          </StatHelpText>
        </Stat>
        
        {/* Accuracy */}
        <Stat bg={statBg} p={4} rounded="md">
          <StatLabel>Accuracy</StatLabel>
          <StatNumber>{(metrics?.accuracy || 0) * 100}%</StatNumber>
          <StatHelpText>
            <Progress
              value={metrics?.accuracy ? metrics.accuracy * 100 : 0}
              size="xs"
              colorScheme="green"
            />
          </StatHelpText>
        </Stat>
        
        {/* Token Count */}
        <Stat bg={statBg} p={4} rounded="md">
          <StatLabel>Tokens Processed</StatLabel>
          <StatNumber>{metrics?.token_count?.toLocaleString() || '-'}</StatNumber>
          <StatHelpText>total tokens</StatHelpText>
        </Stat>
        
        {/* Sequence Length */}
        <Stat bg={statBg} p={4} rounded="md">
          <StatLabel>Sequence Length</StatLabel>
          <StatNumber>{metrics?.sequence_length || '-'}</StatNumber>
          <StatHelpText>tokens/sequence</StatHelpText>
        </Stat>
      </SimpleGrid>
      
      {/* Model Charts */}
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
    </VStack>
  );
};
