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
  HStack,
  Text,
  useColorModeValue,
  Icon
} from '@chakra-ui/react';
import { FaMemory, FaMicrochip, FaHdd, FaNetworkWired, FaThermometerHalf } from 'react-icons/fa';
import { LineChart } from './LineChart';

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

export const SystemPanel = ({ metrics }: { metrics?: SystemMetrics }) => {
  const statBg = useColorModeValue('gray.50', 'gray.600');
  const iconColor = useColorModeValue('blue.500', 'blue.200');
  
  const formatBytes = (bytes: number) => {
    if (!bytes) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`;
  };
  
  return (
    <VStack spacing={6} align="stretch">
      <Heading size="md">System Resources</Heading>
      
      <SimpleGrid columns={{ base: 2, lg: 4 }} spacing={4}>
        {/* CPU Usage */}
        <Stat bg={statBg} p={4} rounded="md">
          <HStack spacing={2}>
            <Icon as={FaMicrochip} color={iconColor} />
            <StatLabel>CPU Usage</StatLabel>
          </HStack>
          <StatNumber>{metrics?.cpu_usage?.toFixed(1)}%</StatNumber>
          <StatHelpText>
            <Progress
              value={metrics?.cpu_usage || 0}
              size="xs"
              colorScheme={metrics?.cpu_usage > 80 ? 'red' : 'green'}
            />
          </StatHelpText>
        </Stat>
        
        {/* Memory Usage */}
        <Stat bg={statBg} p={4} rounded="md">
          <HStack spacing={2}>
            <Icon as={FaMemory} color={iconColor} />
            <StatLabel>Memory Usage</StatLabel>
          </HStack>
          <StatNumber>{metrics?.memory_usage?.toFixed(1)}%</StatNumber>
          <StatHelpText>
            <Progress
              value={metrics?.memory_usage || 0}
              size="xs"
              colorScheme={metrics?.memory_usage > 80 ? 'red' : 'green'}
            />
          </StatHelpText>
        </Stat>
        
        {/* Disk Usage */}
        <Stat bg={statBg} p={4} rounded="md">
          <HStack spacing={2}>
            <Icon as={FaHdd} color={iconColor} />
            <StatLabel>Disk Usage</StatLabel>
          </HStack>
          <StatNumber>{metrics?.disk_usage?.toFixed(1)}%</StatNumber>
          <StatHelpText>
            <Progress
              value={metrics?.disk_usage || 0}
              size="xs"
              colorScheme={metrics?.disk_usage > 80 ? 'red' : 'green'}
            />
          </StatHelpText>
        </Stat>
        
        {/* Network I/O */}
        <Stat bg={statBg} p={4} rounded="md">
          <HStack spacing={2}>
            <Icon as={FaNetworkWired} color={iconColor} />
            <StatLabel>Network I/O</StatLabel>
          </HStack>
          <VStack align="start" spacing={0}>
            <Text fontSize="sm">↑ {formatBytes(metrics?.network_io?.bytes_sent || 0)}</Text>
            <Text fontSize="sm">↓ {formatBytes(metrics?.network_io?.bytes_recv || 0)}</Text>
          </VStack>
        </Stat>
        
        {/* Temperature */}
        {metrics?.temperature !== undefined && (
          <Stat bg={statBg} p={4} rounded="md">
            <HStack spacing={2}>
              <Icon as={FaThermometerHalf} color={iconColor} />
              <StatLabel>Temperature</StatLabel>
            </HStack>
            <StatNumber>{metrics.temperature.toFixed(1)}°C</StatNumber>
            <StatHelpText>
              <Progress
                value={(metrics.temperature / 100) * 100}
                size="xs"
                colorScheme={metrics.temperature > 80 ? 'red' : 'green'}
              />
            </StatHelpText>
          </Stat>
        )}
      </SimpleGrid>
      
      {/* System Charts */}
      <SimpleGrid columns={{ base: 1, lg: 2 }} spacing={4}>
        <LineChart
          title="Resource Usage"
          data={[
            {
              x: new Date(),
              y: metrics?.cpu_usage || 0,
              name: 'CPU'
            },
            {
              x: new Date(),
              y: metrics?.memory_usage || 0,
              name: 'Memory'
            }
          ]}
          yLabel="%"
          showLegend
        />
        <LineChart
          title="Network Traffic"
          data={[
            {
              x: new Date(),
              y: metrics?.network_io?.bytes_sent || 0,
              name: 'Upload'
            },
            {
              x: new Date(),
              y: metrics?.network_io?.bytes_recv || 0,
              name: 'Download'
            }
          ]}
          yLabel="bytes"
          showLegend
        />
      </SimpleGrid>
    </VStack>
  );
};
