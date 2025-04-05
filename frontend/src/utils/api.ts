import { Message } from './types';

// Use a default API URL if environment variable is not set
const API_URL = typeof window !== 'undefined' 
  ? (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000')
  : 'http://localhost:8000';

/**
 * API service for communicating with the backend
 */
export const api = {
  /**
   * Send a message to the chat API and get a response
   */
  async sendMessage(messages: Message[]): Promise<Message> {
    const response = await fetch(`${API_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ messages }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(error || 'Failed to send message');
    }

    return await response.json();
  },

  /**
   * Send a message to the chat API and get a streaming response
   */
  async sendMessageStream(
    messages: Message[],
    onToken: (token: string) => void,
    onComplete: (message: Message) => void,
    onError: (error: Error) => void
  ): Promise<void> {
    try {
      const response = await fetch(`${API_URL}/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ messages }),
      });

      if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to send message');
      }

      // Create a reader for the response body
      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('Response body is not readable');
      }

      // Create a text decoder to convert the binary data to text
      const decoder = new TextDecoder();
      let content = '';

      // Read the stream
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // Decode the received chunk
        const chunk = decoder.decode(value, { stream: true });
        
        // Add the chunk to the content
        content += chunk;
        
        // Notify about new token
        onToken(chunk);
      }

      // Notify about complete message
      onComplete({
        role: 'assistant',
        content
      });
    } catch (error) {
      onError(error instanceof Error ? error : new Error(String(error)));
    }
  },

  /**
   * Check if the API is available
   */
  async health(): Promise<{ status: string }> {
    const response = await fetch(`${API_URL}/health`);
    if (!response.ok) {
      throw new Error('API is not available');
    }
    return await response.json();
  }
};
