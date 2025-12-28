import { useState, useRef, useEffect } from 'react';
import Head from 'next/head';

export default function Home() {
  const [mode, setMode] = useState('smart');
  const [character, setCharacter] = useState('raya');
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [apiUrl, setApiUrl] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [speechSupported, setSpeechSupported] = useState(false);
  const messagesEndRef = useRef(null);
  const recognitionRef = useRef(null);

  useEffect(() => {
    // Set API URL
    setApiUrl(process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000');
    
    // Check for speech recognition support
    if (typeof window !== 'undefined') {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (SpeechRecognition) {
        setSpeechSupported(true);
        const recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = mode === 'dual' && character === 'raya' ? 'ar-AE' : 'en-US';
        
        recognition.onresult = (event) => {
          const transcript = event.results[0][0].transcript;
          setInput(transcript);
          setIsListening(false);
        };
        
        recognition.onerror = (event) => {
          console.error('Speech recognition error:', event.error);
          setIsListening(false);
        };
        
        recognition.onend = () => {
          setIsListening(false);
        };
        
        recognitionRef.current = recognition;
      }
    }
  }, [mode, character]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const startListening = () => {
    if (recognitionRef.current && !isListening) {
      // Update language based on mode/character
      if (mode === 'dual' && character === 'raya') {
        recognitionRef.current.lang = 'ar-AE';
      } else {
        recognitionRef.current.lang = 'en-US';
      }
      
      setIsListening(true);
      recognitionRef.current.start();
    }
  };

  const stopListening = () => {
    if (recognitionRef.current && isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    }
  };

  const playAudio = (audioBase64) => {
    const audio = new Audio(`data:audio/mp3;base64,${audioBase64}`);
    audio.play().catch(err => console.error('Audio playback failed:', err));
  };

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput('');
    setLoading(true);

    const newMessages = [...messages, { role: 'user', content: userMessage }];
    setMessages(newMessages);

    try {
      const response = await fetch(`${apiUrl}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage,
          character: character,
          mode: mode,
          history: newMessages.map(m => ({ role: m.role, content: m.content }))
        })
      });

      if (!response.ok) throw new Error('API request failed');

      const data = await response.json();

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.text_response,
        character: data.character_name,
        emoji: data.emoji,
        audioBase64: data.audio_base64
      }]);

      if (data.audio_base64) {
        playAudio(data.audio_base64);
      }

    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Error: ${error.message}. Please check if the API server is running.`,
        character: 'System',
        emoji: '⚠️'
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <>
      <Head>
        <title>Raya & Mira - Airport Assistants</title>
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1" />
        <meta name="description" content="AI-powered bilingual airport assistants" />
      </Head>

      <div style={styles.container}>
        {/* Header */}
        <div style={styles.header}>
          <h1 style={styles.title}>🎭 Raya & Mira</h1>
          <p style={styles.subtitle}>Your Bilingual Airport Assistants</p>
        </div>

        {/* Mode Selection */}
        <div style={styles.modeContainer}>
          <button
            onClick={() => setMode('smart')}
            style={{...styles.modeButton, ...(mode === 'smart' ? styles.modeButtonActive : {})}}
          >
            🤖 Smart Mode
            <span style={styles.modeDesc}>Auto-detects language</span>
          </button>
          <button
            onClick={() => setMode('dual')}
            style={{...styles.modeButton, ...(mode === 'dual' ? styles.modeButtonActive : {})}}
          >
            👥 Dual Mode
            <span style={styles.modeDesc}>Choose assistant</span>
          </button>
        </div>

        {/* Character Selection (Dual Mode) */}
        {mode === 'dual' && (
          <div style={styles.characterContainer}>
            <button
              onClick={() => setCharacter('raya')}
              style={{...styles.characterButton, ...(character === 'raya' ? styles.characterActive : {})}}
            >
              <span style={styles.charEmoji}>🇦🇪</span>
              <div>
                <div style={styles.charName}>Raya</div>
                <div style={styles.charDesc}>Emirati • Arabic</div>
              </div>
            </button>
            <button
              onClick={() => setCharacter('mira')}
              style={{...styles.characterButton, ...(character === 'mira' ? styles.characterActive : {})}}
            >
              <span style={styles.charEmoji}>🌍</span>
              <div>
                <div style={styles.charName}>Mira</div>
                <div style={styles.charDesc}>International • English</div>
              </div>
            </button>
          </div>
        )}

        {/* Chat Messages */}
        <div style={styles.messagesContainer}>
          {messages.length === 0 && (
            <div style={styles.emptyState}>
              <p style={styles.emptyText}>
                {mode === 'smart' 
                  ? '💬 Type or speak in Arabic or English!' 
                  : `💬 Chat with ${character === 'raya' ? 'Raya 🇦🇪' : 'Mira 🌍'}`}
              </p>
              {speechSupported && (
                <p style={styles.emptySubtext}>
                  🎤 Tap microphone to speak
                </p>
              )}
            </div>
          )}

          {messages.map((msg, idx) => (
            <div
              key={idx}
              style={{
                ...styles.message,
                ...(msg.role === 'user' ? styles.userMessage : styles.assistantMessage)
              }}
            >
              {msg.role === 'assistant' && (
                <div style={styles.messageMeta}>
                  <span style={styles.messageEmoji}>{msg.emoji}</span>
                  <span style={styles.messageName}>{msg.character}</span>
                </div>
              )}
              <div style={styles.messageContent}>{msg.content}</div>
              {msg.audioBase64 && (
                <button
                  onClick={() => playAudio(msg.audioBase64)}
                  style={styles.playButton}
                >
                  🔊 Play Audio
                </button>
              )}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div style={styles.inputContainer}>
          {speechSupported && (
            <button
              onClick={isListening ? stopListening : startListening}
              style={{
                ...styles.micButton,
                ...(isListening ? styles.micButtonActive : {})
              }}
              disabled={loading}
            >
              {isListening ? '🔴' : '🎤'}
            </button>
          )}
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={isListening ? "Listening..." : "Type or speak your message..."}
            style={styles.input}
            rows={2}
            disabled={loading || isListening}
          />
          <button
            onClick={sendMessage}
            disabled={loading || !input.trim() || isListening}
            style={{
              ...styles.sendButton,
              ...(loading || !input.trim() || isListening ? styles.sendButtonDisabled : {})
            }}
          >
            {loading ? '⏳' : '📤'}
          </button>
        </div>

        {/* Footer */}
        <div style={styles.footer}>
          {speechSupported ? '🎤 Voice enabled • ' : ''}Powered by Groq • Edge TTS • Built by Mr. Data
        </div>
      </div>
    </>
  );
}

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    maxWidth: '800px',
    margin: '0 auto',
    backgroundColor: '#f5f5f5',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
  },
  header: {
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: 'white',
    padding: '20px',
    textAlign: 'center',
    boxShadow: '0 2px 10px rgba(0,0,0,0.1)'
  },
  title: {
    margin: '0 0 5px 0',
    fontSize: '24px',
    fontWeight: 'bold'
  },
  subtitle: {
    margin: 0,
    fontSize: '14px',
    opacity: 0.9
  },
  modeContainer: {
    display: 'flex',
    gap: '10px',
    padding: '15px',
    backgroundColor: 'white',
    borderBottom: '1px solid #e0e0e0'
  },
  modeButton: {
    flex: 1,
    padding: '12px',
    border: '2px solid #e0e0e0',
    borderRadius: '10px',
    backgroundColor: 'white',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: '600',
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
    transition: 'all 0.2s'
  },
  modeButtonActive: {
    borderColor: '#667eea',
    backgroundColor: '#f0f4ff'
  },
  modeDesc: {
    fontSize: '11px',
    fontWeight: 'normal',
    color: '#666'
  },
  characterContainer: {
    display: 'flex',
    gap: '10px',
    padding: '15px',
    backgroundColor: 'white',
    borderBottom: '1px solid #e0e0e0'
  },
  characterButton: {
    flex: 1,
    padding: '15px',
    border: '2px solid #e0e0e0',
    borderRadius: '10px',
    backgroundColor: 'white',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    transition: 'all 0.2s'
  },
  characterActive: {
    borderColor: '#667eea',
    backgroundColor: '#f0f4ff'
  },
  charEmoji: {
    fontSize: '32px'
  },
  charName: {
    fontSize: '16px',
    fontWeight: '600',
    marginBottom: '2px'
  },
  charDesc: {
    fontSize: '12px',
    color: '#666'
  },
  messagesContainer: {
    flex: 1,
    overflowY: 'auto',
    padding: '20px',
    display: 'flex',
    flexDirection: 'column',
    gap: '15px'
  },
  emptyState: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100%',
    color: '#999'
  },
  emptyText: {
    fontSize: '16px',
    marginBottom: '5px'
  },
  emptySubtext: {
    fontSize: '14px',
    color: '#bbb'
  },
  message: {
    maxWidth: '80%',
    padding: '12px 16px',
    borderRadius: '18px',
    wordWrap: 'break-word'
  },
  userMessage: {
    alignSelf: 'flex-end',
    backgroundColor: '#667eea',
    color: 'white'
  },
  assistantMessage: {
    alignSelf: 'flex-start',
    backgroundColor: 'white',
    boxShadow: '0 1px 2px rgba(0,0,0,0.1)'
  },
  messageMeta: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    marginBottom: '6px',
    fontSize: '13px',
    fontWeight: '600',
    color: '#667eea'
  },
  messageEmoji: {
    fontSize: '16px'
  },
  messageName: {},
  messageContent: {
    lineHeight: '1.5'
  },
  playButton: {
    marginTop: '8px',
    padding: '6px 12px',
    border: 'none',
    borderRadius: '12px',
    backgroundColor: '#f0f4ff',
    color: '#667eea',
    fontSize: '12px',
    fontWeight: '600',
    cursor: 'pointer'
  },
  inputContainer: {
    display: 'flex',
    gap: '10px',
    padding: '15px',
    backgroundColor: 'white',
    borderTop: '1px solid #e0e0e0'
  },
  micButton: {
    padding: '12px 16px',
    border: '2px solid #e0e0e0',
    borderRadius: '12px',
    backgroundColor: 'white',
    fontSize: '20px',
    cursor: 'pointer',
    transition: 'all 0.2s'
  },
  micButtonActive: {
    backgroundColor: '#ff4444',
    borderColor: '#ff4444',
    animation: 'pulse 1s infinite'
  },
  input: {
    flex: 1,
    padding: '12px',
    border: '2px solid #e0e0e0',
    borderRadius: '12px',
    fontSize: '15px',
    resize: 'none',
    fontFamily: 'inherit'
  },
  sendButton: {
    padding: '12px 20px',
    border: 'none',
    borderRadius: '12px',
    backgroundColor: '#667eea',
    color: 'white',
    fontSize: '20px',
    cursor: 'pointer',
    transition: 'all 0.2s'
  },
  sendButtonDisabled: {
    backgroundColor: '#ccc',
    cursor: 'not-allowed'
  },
  footer: {
    padding: '10px',
    textAlign: 'center',
    fontSize: '11px',
    color: '#999',
    backgroundColor: 'white',
    borderTop: '1px solid #e0e0e0'
  }
};
