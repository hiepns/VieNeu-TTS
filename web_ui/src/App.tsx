// src/App.tsx

import { useState, useEffect } from 'react';
import { Mic, Square, Loader2, AlertCircle } from 'lucide-react';
import { useWebSocket } from './hooks/useWebSocket';
import { useAudioPlayer } from './hooks/useAudioPlayer';
import { AudioVisualizer } from './components/AudioVisualizer';
import { Voice, WSMessage } from './types';

const VOICES: Voice[] = [
  { id: 'vinh', name: 'Vĩnh (nam miền Nam)', region: 'south', gender: 'male', audioPath: '', textPath: '' },
  { id: 'binh', name: 'Bình (nam miền Bắc)', region: 'north', gender: 'male', audioPath: '', textPath: '' },
  { id: 'ngoc', name: 'Ngọc (nữ miền Bắc)', region: 'north', gender: 'female', audioPath: '', textPath: '' },
  { id: 'dung', name: 'Dung (nữ miền Nam)', region: 'south', gender: 'female', audioPath: '', textPath: '' },
];

const WS_URL = 'ws://localhost:8000/ws/tts';

function App() {
  const [text, setText] = useState('Hà Nội, trái tim của Việt Nam, là một thành phố ngàn năm văn hiến với bề dày lịch sử và văn hóa độc đáo.');
  const [selectedVoice, setSelectedVoice] = useState<string>(VOICES[0].id);
  const [status, setStatus] = useState('⏳ Sẵn sàng');

  const { connect, send, disconnect, state, error, onMessage } = useWebSocket(WS_URL);
  const { initialize, enqueue, stop, isPlaying, analyser } = useAudioPlayer();

  useEffect(() => {
    onMessage((msg: WSMessage) => {
      if (msg.type === 'metadata') {
        initialize(msg.sample_rate || 24000);
        setStatus('🎵 Đang tổng hợp...');
      } else if (msg.type === 'audio_chunk' && msg.data) {
        const audioData = base64ToFloat32(msg.data);
        enqueue(audioData);
        setStatus('🔊 Đang phát...');
      } else if (msg.type === 'end') {
        setStatus('✅ Hoàn tất!');
      } else if (msg.type === 'error') {
        setStatus(`❌ Lỗi: ${msg.message}`);
      }
    });
  }, [onMessage, initialize, enqueue]);

  const base64ToFloat32 = (base64: string): Float32Array => {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    return new Float32Array(bytes.buffer);
  };

  const handleGenerate = async () => {
    if (!text.trim()) {
      alert('Vui lòng nhập văn bản!');
      return;
    }

    try {
      setStatus('🔌 Đang kết nối...');
      await connect();
      send({
        text: text.trim(),
        voice: selectedVoice,
        mode: 'streaming',
      });
    } catch (e) {
      setStatus('❌ Không thể kết nối tới server');
      console.error(e);
    }
  };

  const handleStop = () => {
    stop();
    disconnect();
    setStatus('⏹️ Đã dừng');
  };

  const isGenerating = state === 'connecting' || state === 'playing';

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white p-6">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="text-center mb-10">
          <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent mb-2">
            🦜 VieNeu-TTS Studio
          </h1>
          <p className="text-slate-400">Real-time Vietnamese Text-to-Speech</p>
        </div>

        {/* Main Card */}
        <div className="bg-slate-800/50 backdrop-blur-xl rounded-2xl p-8 shadow-2xl border border-slate-700">
          {/* Text Input */}
          <div className="mb-6">
            <label className="block text-sm font-medium mb-2 text-slate-300">
              📝 Văn bản
            </label>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              className="w-full bg-slate-900/60 border border-slate-600 rounded-lg p-4 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
              rows={6}
              placeholder="Nhập văn bản cần tổng hợp..."
              disabled={isGenerating}
            />
            <div className="text-right text-sm text-slate-400 mt-1">
              {text.length} ký tự
            </div>
          </div>

          {/* Voice Selector */}
          <div className="mb-6">
            <label className="block text-sm font-medium mb-2 text-slate-300">
              🎤 Giọng đọc
            </label>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {VOICES.map((voice) => (
                <button
                  key={voice.id}
                  onClick={() => setSelectedVoice(voice.id)}
                  disabled={isGenerating}
                  className={`p-3 rounded-lg border-2 transition-all ${
                    selectedVoice === voice.id
                      ? 'bg-gradient-to-r from-blue-500 to-cyan-500 border-blue-400 shadow-lg shadow-blue-500/50'
                      : 'bg-slate-900/60 border-slate-600 hover:border-slate-500'
                  } disabled:opacity-50 disabled:cursor-not-allowed`}
                >
                  <div className="text-sm font-medium">{voice.name.split(' ')[0]}</div>
                  <div className="text-xs text-slate-400 mt-1">
                    {voice.region === 'north' ? 'Miền Bắc' : 'Miền Nam'}
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Visualizer */}
          <div className="mb-6">
            <AudioVisualizer analyser={analyser} isPlaying={isPlaying} />
          </div>

          {/* Controls */}
          <div className="flex gap-4 mb-4">
            {!isGenerating ? (
              <button
                onClick={handleGenerate}
                className="flex-1 bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 text-white font-semibold py-4 px-6 rounded-lg transition-all transform hover:scale-105 hover:shadow-lg hover:shadow-blue-500/50 flex items-center justify-center gap-2"
              >
                <Mic className="w-5 h-5" />
                Bắt đầu tổng hợp
              </button>
            ) : (
              <button
                onClick={handleStop}
                className="flex-1 bg-gradient-to-r from-red-500 to-pink-500 hover:from-red-600 hover:to-pink-600 text-white font-semibold py-4 px-6 rounded-lg transition-all transform hover:scale-105 hover:shadow-lg hover:shadow-red-500/50 flex items-center justify-center gap-2"
              >
                <Square className="w-5 h-5" />
                Dừng lại
              </button>
            )}
          </div>

          {/* Status */}
          <div className={`p-4 rounded-lg border-l-4 ${
            error 
              ? 'bg-red-500/10 border-red-500 text-red-300'
              : state === 'playing'
              ? 'bg-green-500/10 border-green-500 text-green-300'
              : 'bg-cyan-500/10 border-cyan-500 text-cyan-300'
          }`}>
            <div className="flex items-center gap-2">
              {state === 'connecting' && <Loader2 className="w-4 h-4 animate-spin" />}
              {error && <AlertCircle className="w-4 h-4" />}
              <span className="font-medium">{status}</span>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-8 text-slate-400 text-sm">
          Powered by VieNeu-TTS • WebSocket Streaming
        </div>
      </div>
    </div>
  );
}

export default App;