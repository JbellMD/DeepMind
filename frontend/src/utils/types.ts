/**
 * Types for the DeepMind chat application
 */

/**
 * Message interface for chat messages
 */
export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: number;
}

/**
 * Chat session interface for managing chat history
 */
export interface ChatSession {
  id: string;
  name: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
  onUpdateMessages?: (messages: Message[]) => void;
}

/**
 * Chat API request interface
 */
export interface ChatRequest {
  messages: Message[];
  temperature?: number;
  max_tokens?: number;
}

/**
 * Health check response
 */
export interface HealthResponse {
  status: string;
  model?: string;
  version?: string;
}
