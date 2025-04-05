import { useState, useRef, useEffect } from 'react';
import { FiSend, FiMenu } from 'react-icons/fi';
import TextareaAutosize from 'react-textarea-autosize';
import { ChatSession, Message } from '@/utils/types';
import ChatMessage from '@/components/ChatMessage';
import { api } from '@/utils/api';

interface ChatContainerProps {
  session: ChatSession;
  onSidebarToggle: () => void;
}

const ChatContainer: React.FC<ChatContainerProps> = ({ session, onSidebarToggle }) => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>(session.messages || []);
  const [isLoading, setIsLoading] = useState(false);
  const [streamingContent, setStreamingContent] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  
  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingContent]);

  // Auto focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Update local messages when session messages change
  useEffect(() => {
    setMessages(session.messages || []);
  }, [session.id]);
  
  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!input.trim() || isLoading) return;
    
    // Create user message
    const userMessage: Message = {
      role: 'user',
      content: input.trim(),
    };
    
    // Update UI immediately with user message
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setInput('');
    
    // Set loading state
    setIsLoading(true);
    setStreamingContent('');
    
    try {
      // Send message to API with streaming response
      await api.sendMessageStream(
        updatedMessages,
        (token) => {
          // Handle each token as it arrives
          setStreamingContent((prev) => prev + token);
        },
        (completeMessage) => {
          // Handle complete message
          setMessages([...updatedMessages, completeMessage]);
          setStreamingContent('');
          setIsLoading(false);
          // Update session messages
          if (session.onUpdateMessages) {
            session.onUpdateMessages([...updatedMessages, completeMessage]);
          }
        },
        (error) => {
          // Handle error
          console.error('Error sending message:', error);
          setIsLoading(false);
          
          // Add error message
          const errorMessage: Message = {
            role: 'assistant',
            content: `Sorry, there was an error processing your request: ${error.message}`,
          };
          setMessages([...updatedMessages, errorMessage]);
          
          // Update session messages
          if (session.onUpdateMessages) {
            session.onUpdateMessages([...updatedMessages, errorMessage]);
          }
        }
      );
    } catch (error) {
      console.error('Error sending message:', error);
      setIsLoading(false);
    }
  };
  
  // Handle input change
  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
  };
  
  // Handle key press (Ctrl+Enter to submit)
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };
  
  return (
    <div className="flex flex-col h-full">
      {/* Mobile header */}
      <div className="lg:hidden flex items-center p-4 border-b border-[rgba(var(--border-rgb),0.1)]">
        <button 
          onClick={onSidebarToggle}
          className="p-2 rounded-md hover:bg-gray-100 dark:hover:bg-gray-800"
        >
          <FiMenu className="text-gray-600 dark:text-gray-300" />
        </button>
        <div className="ml-4 font-medium truncate">{session.name}</div>
      </div>
      
      {/* Message container */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center text-gray-500 dark:text-gray-400 p-8">
            <h2 className="text-2xl font-bold mb-2">Welcome to DeepMind AI</h2>
            <p className="mb-4">
              I'm your advanced AI assistant. How can I help you today?
            </p>
          </div>
        ) : (
          <>
            {/* Render all messages */}
            {messages.map((message, index) => (
              <ChatMessage key={index} message={message} />
            ))}
            
            {/* Render streaming message if any */}
            {streamingContent && (
              <ChatMessage 
                message={{ 
                  role: 'assistant', 
                  content: streamingContent 
                }} 
                isStreaming={true}
              />
            )}
          </>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      {/* Input area */}
      <div className="border-t border-[rgba(var(--border-rgb),0.1)] p-4">
        <form onSubmit={handleSubmit} className="relative">
          <TextareaAutosize
            ref={inputRef}
            value={input}
            onChange={handleInputChange}
            onKeyDown={handleKeyPress}
            placeholder="Send a message..."
            className="w-full p-3 pr-12 rounded-lg border border-[rgba(var(--border-rgb),0.2)] bg-[var(--input-bg)] text-[var(--text-color)] resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400 min-h-[56px] max-h-[200px]"
            maxRows={5}
            disabled={isLoading}
          />
          <button
            type="submit"
            className={`absolute right-3 bottom-[12px] p-2 rounded-md ${
              !input.trim() || isLoading
                ? 'text-gray-400 dark:text-gray-600 cursor-not-allowed'
                : 'text-blue-500 hover:text-blue-600 dark:hover:text-blue-400'
            }`}
            disabled={!input.trim() || isLoading}
          >
            <FiSend />
          </button>
        </form>
        <div className="mt-2 text-xs text-center text-gray-400">
          Press Enter to send, Shift+Enter for new line
        </div>
      </div>
    </div>
  );
};

export default ChatContainer;
