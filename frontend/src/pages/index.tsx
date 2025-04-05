import React, { useState, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { AnimatePresence, motion } from 'framer-motion';
import { FiMenu } from 'react-icons/fi';
import ChatSidebar from '@/components/ChatSidebar';
import ChatContainer from '@/components/ChatContainer';
import { ChatSession, Message } from '@/utils/types';

const HomePage: React.FC = () => {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string>('');
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);

  // Initialize sessions from localStorage or create a new one
  useEffect(() => {
    const savedSessions = localStorage.getItem('chatSessions');
    let initialSessions: ChatSession[] = [];
    
    if (savedSessions) {
      try {
        const parsedSessions = JSON.parse(savedSessions);
        if (Array.isArray(parsedSessions) && parsedSessions.length > 0) {
          initialSessions = parsedSessions;
        } else {
          initialSessions = [createNewSession()];
        }
      } catch (error) {
        console.error('Error parsing saved sessions:', error);
        initialSessions = [createNewSession()];
      }
    } else {
      initialSessions = [createNewSession()];
    }
    
    setSessions(initialSessions);
    setActiveSessionId(initialSessions[0].id);
  }, []);

  // Save sessions to localStorage when they change
  useEffect(() => {
    if (sessions.length > 0) {
      localStorage.setItem('chatSessions', JSON.stringify(sessions));
    }
  }, [sessions]);

  // Create a new chat session
  const createNewSession = (): ChatSession => {
    return {
      id: uuidv4(),
      name: `New Chat ${new Date().toLocaleString('en-US', { 
        month: 'short', 
        day: 'numeric',
        hour: 'numeric',
        minute: 'numeric'
      })}`,
      messages: [],
      createdAt: Date.now(),
      updatedAt: Date.now()
    };
  };

  // Add a new session
  const handleCreateSession = () => {
    const newSession = createNewSession();
    setSessions([newSession, ...sessions]);
    setActiveSessionId(newSession.id);
    setIsMobileSidebarOpen(false);
  };

  // Rename a session
  const handleRenameSession = (sessionId: string, newName: string) => {
    setSessions(
      sessions.map(session => 
        session.id === sessionId 
          ? { ...session, name: newName, updatedAt: Date.now() } 
          : session
      )
    );
  };

  // Delete a session
  const handleDeleteSession = (sessionId: string) => {
    const newSessions = sessions.filter(session => session.id !== sessionId);
    setSessions(newSessions);
    
    // If the active session was deleted, set a new active session
    if (sessionId === activeSessionId && newSessions.length > 0) {
      setActiveSessionId(newSessions[0].id);
    } else if (newSessions.length === 0) {
      // If all sessions were deleted, create a new one
      const newSession = createNewSession();
      setSessions([newSession]);
      setActiveSessionId(newSession.id);
    }
  };

  // Update messages in a session
  const handleUpdateMessages = (sessionId: string, messages: Message[]) => {
    setSessions(
      sessions.map(session =>
        session.id === sessionId
          ? { ...session, messages, updatedAt: Date.now() }
          : session
      )
    );
  };

  // Get the active session
  const activeSession = sessions.find(session => session.id === activeSessionId) || sessions[0] || createNewSession();
  
  // Add update function to the active session
  const sessionWithUpdater = {
    ...activeSession,
    onUpdateMessages: (messages: Message[]) => handleUpdateMessages(activeSession.id, messages)
  };

  return (
    <div className="flex h-screen bg-[var(--bg-color)] text-[var(--text-color)]">
      {/* Mobile Overlay */}
      <AnimatePresence>
        {isMobileSidebarOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.5 }}
            exit={{ opacity: 0 }}
            className="lg:hidden fixed inset-0 z-10 bg-black"
            onClick={() => setIsMobileSidebarOpen(false)}
          />
        )}
      </AnimatePresence>
      
      {/* Sidebar for chat sessions */}
      <ChatSidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={(id) => {
          setActiveSessionId(id);
          setIsMobileSidebarOpen(false);
        }}
        onCreateSession={handleCreateSession}
        onRenameSession={handleRenameSession}
        onDeleteSession={handleDeleteSession}
        isMobileOpen={isMobileSidebarOpen}
        onCloseMobile={() => setIsMobileSidebarOpen(false)}
      />
      
      {/* Main Chat Container */}
      <div className="flex-1 flex flex-col h-full overflow-hidden">
        {/* Mobile Header */}
        <div className="lg:hidden flex items-center p-4 border-b border-[rgba(var(--border-rgb),0.1)]">
          <button
            onClick={() => setIsMobileSidebarOpen(true)}
            className="p-2 rounded-md hover:bg-gray-100 dark:hover:bg-gray-800"
          >
            <FiMenu className="text-[var(--text-color)]" />
          </button>
          <h1 className="ml-2 font-semibold">DeepMind AI</h1>
        </div>
        
        {/* Chat Messages and Input */}
        <div className="flex-1 overflow-hidden">
          <ChatContainer
            session={sessionWithUpdater}
            onSidebarToggle={() => setIsMobileSidebarOpen(true)}
          />
        </div>
      </div>
    </div>
  );
};

export default HomePage;
