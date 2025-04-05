import React, { createContext, useState, useContext, useEffect } from 'react';

type Theme = 'light' | 'dark';

interface ThemeContextType {
  theme: Theme;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextType>({
  theme: 'dark',
  toggleTheme: () => {},
});

export const useTheme = () => useContext(ThemeContext);

export const ThemeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [theme, setTheme] = useState<Theme>('dark');

  // On mount, check if there's a theme preference in localStorage
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme') as Theme;
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    if (savedTheme) {
      setTheme(savedTheme);
    } else if (prefersDark) {
      setTheme('dark');
    }
  }, []);

  // Update document with current theme
  useEffect(() => {
    const root = document.documentElement;
    
    if (theme === 'dark') {
      root.classList.add('dark');
      // Set CSS variables for dark mode
      root.style.setProperty('--bg-color', '#111827');
      root.style.setProperty('--text-color', '#f3f4f6');
      root.style.setProperty('--border-rgb', '75, 85, 99');
      root.style.setProperty('--message-user-bg', '#1f2937');
      root.style.setProperty('--message-bot-bg', '#1a2133');
      root.style.setProperty('--input-bg', '#1f2937');
      root.style.setProperty('--primary-button-bg', '#3b82f6');
      root.style.setProperty('--primary-button-hover-bg', '#2563eb');
      root.style.setProperty('--secondary-button-bg', '#374151');
      root.style.setProperty('--secondary-button-hover-bg', '#4b5563');
      root.style.setProperty('--active-session-bg', '#2563eb');
      root.style.setProperty('--active-session-text', '#ffffff');
      root.style.setProperty('--hover-session-bg', '#1f2937');
    } else {
      root.classList.remove('dark');
      // Set CSS variables for light mode
      root.style.setProperty('--bg-color', '#ffffff');
      root.style.setProperty('--text-color', '#111827');
      root.style.setProperty('--border-rgb', '229, 231, 235');
      root.style.setProperty('--message-user-bg', '#f3f4f6');
      root.style.setProperty('--message-bot-bg', '#f9fafb');
      root.style.setProperty('--input-bg', '#ffffff');
      root.style.setProperty('--primary-button-bg', '#3b82f6');
      root.style.setProperty('--primary-button-hover-bg', '#2563eb');
      root.style.setProperty('--secondary-button-bg', '#f3f4f6');
      root.style.setProperty('--secondary-button-hover-bg', '#e5e7eb');
      root.style.setProperty('--active-session-bg', '#dbeafe');
      root.style.setProperty('--active-session-text', '#1e40af');
      root.style.setProperty('--hover-session-bg', '#f3f4f6');
    }
    
    // Save theme preference to localStorage
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prevTheme => (prevTheme === 'dark' ? 'light' : 'dark'));
  };

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};
