// pages/index.tsx
'use client';

import { useState, useRef, useEffect } from 'react';
import { Loader2, Send, Upload, TrendingUp } from 'lucide-react';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: string[];
  timestamp: Date;
}

export default function TradingAdvisor() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [documents, setDocuments] = useState<string[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Send message
  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!input.trim()) return;

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: input,
          conversation_history: messages.map(m => ({
            role: m.role,
            content: m.content
          }))
        }),
      });

      const data = await response.json();

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.response || 'No response received',
        sources: data.sources || [],
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Error: Could not reach the server. Make sure the backend is running.',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  // Handle document upload
  const handleDocumentUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    for (const file of files) {
      const formData = new FormData();
      formData.append('file', file);

      try {
        setUploadProgress(50);
        
        const response = await fetch('http://localhost:8000/documents/upload', {
          method: 'POST',
          body: formData,
        });

        const data = await response.json();

        setDocuments(prev => [...prev, file.name]);
        setUploadProgress(100);

        // Reset
        setTimeout(() => setUploadProgress(0), 2000);
      } catch (error) {
        console.error('Upload failed:', error);
        setUploadProgress(0);
      }
    }
  };

  // Analyze stock
  const handleStockAnalysis = async (ticker: string) => {
    setInput(ticker);
    
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/analysis/stock', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ticker: ticker.toUpperCase(),
          analysis_type: 'full'
        }),
      });

      const data = await response.json();

      const analysisMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Stock Analysis for ${ticker}:\n\nPrice: $${data.price}\n\nTechnical:\n- Total Return: ${data.technical?.returns?.total_return?.toFixed(2)}%\n- RSI: ${data.technical?.rsi?.toFixed(2)}\n- Signal: ${data.technical?.signal}`,
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, analysisMessage]);
    } catch (error) {
      console.error('Analysis failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex flex-col">
      {/* Header */}
      <div className="bg-slate-950 border-b border-slate-700 p-4 shadow-lg">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center gap-3 mb-2">
            <TrendingUp className="w-6 h-6 text-emerald-500" />
            <h1 className="text-2xl font-bold text-white">AI Trading Advisor</h1>
          </div>
          <p className="text-slate-400">RAG-powered financial research and insights</p>
        </div>
      </div>

      {/* Documents Section */}
      <div className="bg-slate-800 border-b border-slate-700 p-4">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center gap-4">
            <button
              onClick={() => fileInputRef.current?.click()}
              className="flex items-center gap-2 bg-emerald-600 hover:bg-emerald-700 text-white px-4 py-2 rounded-lg transition"
            >
              <Upload className="w-4 h-4" />
              Upload Documents
            </button>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".pdf,.txt,.docx"
              onChange={handleDocumentUpload}
              className="hidden"
            />
            
            {uploadProgress > 0 && (
              <div className="flex-1">
                <div className="w-full bg-slate-700 rounded-full h-2">
                  <div
                    className="bg-emerald-500 h-2 rounded-full transition-all"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
              </div>
            )}

            {documents.length > 0 && (
              <div className="text-sm text-slate-400">
                {documents.length} document(s) uploaded
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="max-w-4xl mx-auto space-y-4">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center">
              <TrendingUp className="w-16 h-16 text-slate-600 mb-4" />
              <h2 className="text-2xl font-bold text-slate-300 mb-2">Welcome to AI Trading Advisor</h2>
              <p className="text-slate-400 mb-6">Upload documents and ask questions about financial markets</p>
              
              <div className="grid grid-cols-3 gap-4 mt-8">
                <button
                  onClick={() => handleStockAnalysis('AAPL')}
                  className="bg-slate-700 hover:bg-slate-600 text-white px-4 py-3 rounded-lg transition"
                >
                  Analyze AAPL
                </button>
                <button
                  onClick={() => handleStockAnalysis('MSFT')}
                  className="bg-slate-700 hover:bg-slate-600 text-white px-4 py-3 rounded-lg transition"
                >
                  Analyze MSFT
                </button>
                <button
                  onClick={() => handleStockAnalysis('TSLA')}
                  className="bg-slate-700 hover:bg-slate-600 text-white px-4 py-3 rounded-lg transition"
                >
                  Analyze TSLA
                </button>
              </div>
            </div>
          ) : (
            messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-2xl rounded-lg p-4 ${
                    message.role === 'user'
                      ? 'bg-emerald-600 text-white'
                      : 'bg-slate-700 text-slate-100'
                  }`}
                >
                  <p className="whitespace-pre-wrap">{message.content}</p>
                  {message.sources && message.sources.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-slate-600">
                      <p className="text-xs font-semibold mb-2">Sources:</p>
                      <ul className="text-xs space-y-1">
                        {message.sources.map((source, idx) => (
                          <li key={idx} className="text-slate-300">• {source.substring(0, 50)}...</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  <p className="text-xs mt-2 opacity-70">
                    {message.timestamp.toLocaleTimeString()}
                  </p>
                </div>
              </div>
            ))
          )}
          
          {loading && (
            <div className="flex justify-start">
              <div className="bg-slate-700 text-slate-100 rounded-lg p-4 flex items-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Analyzing...</span>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="bg-slate-800 border-t border-slate-700 p-4">
        <div className="max-w-4xl mx-auto">
          <form onSubmit={handleSendMessage} className="flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about stocks, markets, or uploaded documents..."
              className="flex-1 bg-slate-700 text-white placeholder-slate-400 px-4 py-3 rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-500"
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-600 text-white px-6 py-3 rounded-lg transition flex items-center gap-2"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
              Send
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
