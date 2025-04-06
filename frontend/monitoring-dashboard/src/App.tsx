import React, { useEffect, useState } from 'react';
import {
  ChakraProvider,
  Box,
  Grid,
  theme,
  VStack,
  Heading,
  Text,
  Flex,
  useColorModeValue,
  IconButton
} from '@chakra-ui/react';
import { FaSun, FaMoon } from 'react-icons/fa';
import { PerformancePanel } from './components/PerformancePanel';
import { ModelPanel } from './components/ModelPanel';
import { SystemPanel } from './components/SystemPanel';
import { useWebSocket } from './hooks/useWebSocket';
import { useMetrics } from './hooks/useMetrics';

export const App = () => {
  const { metrics, updateMetrics } = useMetrics();
  
  // WebSocket connection
  useWebSocket('ws://localhost:8000/ws', {
    onMessage: (data) => updateMetrics(JSON.parse(data))
  });
  
  // Color mode
  const bg = useColorModeValue('gray.50', 'gray.800');
  const color = useColorModeValue('gray.800', 'white');
  
  return (
    <ChakraProvider theme={theme}>
      <Box minH="100vh" bg={bg} color={color} p={4}>
        <VStack spacing={8} w="full">
          {/* Header */}
          <Flex w="full" justify="space-between" align="center">
            <Heading size="lg">DeepMind Monitoring Dashboard</Heading>
            <IconButton
              aria-label="Toggle color mode"
              icon={useColorModeValue(<FaMoon />, <FaSun />)}
              onClick={() => {}}
            />
          </Flex>
          
          {/* Dashboard Grid */}
          <Grid
            templateColumns={{ base: '1fr', lg: 'repeat(2, 1fr)' }}
            gap={6}
            w="full"
          >
            {/* Performance Metrics */}
            <Box
              bg={useColorModeValue('white', 'gray.700')}
              p={6}
              rounded="lg"
              shadow="base"
            >
              <PerformancePanel metrics={metrics.performance} />
            </Box>
            
            {/* Model Metrics */}
            <Box
              bg={useColorModeValue('white', 'gray.700')}
              p={6}
              rounded="lg"
              shadow="base"
            >
              <ModelPanel metrics={metrics.model} />
            </Box>
            
            {/* System Metrics */}
            <Box
              bg={useColorModeValue('white', 'gray.700')}
              p={6}
              rounded="lg"
              shadow="base"
              gridColumn={{ lg: 'span 2' }}
            >
              <SystemPanel metrics={metrics.system} />
            </Box>
          </Grid>
        </VStack>
      </Box>
    </ChakraProvider>
  );
};
