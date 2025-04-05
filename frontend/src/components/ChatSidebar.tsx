import React, { useState } from 'react';
import { FiPlus, FiEdit2, FiTrash2, FiMoon, FiSun, FiX } from 'react-icons/fi';
import { useTheme } from '@/utils/ThemeContext';
import { ChatSession } from '@/utils/types';

interface ChatSidebarProps {
  sessions: ChatSession[];
  activeSessionId: string;
  onSelectSession: (sessionId: string) => void;
  onCreateSession: () => void;
  onRenameSession: (sessionId: string, name: string) => void;
  onDeleteSession: (sessionId: string) => void;
  isMobileOpen: boolean;
  onCloseMobile: () => void;
}

const ChatSidebar: React.FC<ChatSidebarProps> = ({
  sessions,
  activeSessionId,
  onSelectSession,
  onCreateSession,
  onRenameSession,
  onDeleteSession,
  isMobileOpen,
  onCloseMobile,
}) => {
  const { theme, toggleTheme } = useTheme();
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [newName, setNewName] = useState('');

  // Start renaming a session
  const handleRenameClick = (sessionId: string, currentName: string) => {
    setRenamingId(sessionId);
    setNewName(currentName);
  };

  // Save the new name
  const handleSaveRename = (sessionId: string) => {
    if (newName.trim()) {
      onRenameSession(sessionId, newName.trim());
    }
    setRenamingId(null);
    setNewName('');
  };

  // Handle key press during rename (Enter to save, Escape to cancel)
  const handleRenameKeyPress = (e: React.KeyboardEvent, sessionId: string) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSaveRename(sessionId);
    } else if (e.key === 'Escape') {
      setRenamingId(null);
      setNewName('');
    }
  };

  return (
    <div
      className={`fixed lg:relative inset-0 transform ${
        isMobileOpen ? 'translate-x-0' : '-translate-x-full'
      } lg:translate-x-0 z-10 w-full lg:w-80 bg-gray-50 dark:bg-gray-900 border-r border-[rgba(var(--border-rgb),0.1)] transition-transform duration-300 ease-in-out flex flex-col h-full`}
    >
      {/* Mobile Close Button */}
      <div className="lg:hidden flex justify-end p-2">
        <button
          onClick={onCloseMobile}
          className="p-2 rounded-full hover:bg-gray-200 dark:hover:bg-gray-800"
        >
          <FiX className="text-gray-500 dark:text-gray-400" />
        </button>
      </div>

      {/* Header */}
      <div className="p-4 border-b border-[rgba(var(--border-rgb),0.1)]">
        <h1 className="text-xl font-bold text-[var(--text-color)]">DeepMind AI</h1>
      </div>

      {/* New Chat Button */}
      <div className="p-3">
        <button
          onClick={onCreateSession}
          className="flex items-center justify-center w-full py-2 px-4 bg-[var(--primary-button-bg)] hover:bg-[var(--primary-button-hover-bg)] text-white rounded-lg transition-colors"
        >
          <FiPlus className="mr-2" /> New Chat
        </button>
      </div>

      {/* Sessions List */}
      <div className="flex-1 overflow-y-auto py-2 px-3 space-y-1">
        {sessions.map((session) => (
          <div
            key={session.id}
            className={`rounded-lg transition-colors ${
              session.id === activeSessionId
                ? 'bg-[var(--active-session-bg)] text-[var(--active-session-text)]'
                : 'hover:bg-[var(--hover-session-bg)] text-[var(--text-color)]'
            }`}
          >
            {renamingId === session.id ? (
              <div className="flex p-2">
                <input
                  type="text"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  onKeyDown={(e) => handleRenameKeyPress(e, session.id)}
                  onBlur={() => handleSaveRename(session.id)}
                  autoFocus
                  className="flex-1 px-2 py-1 bg-white dark:bg-gray-800 border border-blue-500 dark:border-blue-400 rounded focus:outline-none"
                />
              </div>
            ) : (
              <div className="flex items-center p-3">
                <div
                  className="flex-1 truncate cursor-pointer"
                  onClick={() => onSelectSession(session.id)}
                >
                  {session.name}
                </div>
                <div className="flex space-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleRenameClick(session.id, session.name);
                    }}
                    className="p-1 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 rounded"
                  >
                    <FiEdit2 size={16} />
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteSession(session.id);
                    }}
                    className="p-1 text-gray-500 dark:text-gray-400 hover:text-red-600 dark:hover:text-red-400 rounded"
                  >
                    <FiTrash2 size={16} />
                  </button>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-[rgba(var(--border-rgb),0.1)]">
        <button
          onClick={toggleTheme}
          className="flex items-center justify-center w-full py-2 px-4 bg-[var(--secondary-button-bg)] hover:bg-[var(--secondary-button-hover-bg)] text-[var(--text-color)] rounded-lg transition-colors"
        >
          {theme === 'dark' ? (
            <>
              <FiSun className="mr-2" /> Light Mode
            </>
          ) : (
            <>
              <FiMoon className="mr-2" /> Dark Mode
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default ChatSidebar;
