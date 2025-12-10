// src/hooks/useWebSocket.ts

import { useState, useRef, useCallback } from 'react';
import { WSMessage, TTSRequest, PlaybackState } from '../types';

interface UseWebSocketReturn {
  connect: () => Promise<void>;
  send: (request: TTSRequest) => void;
  disconnect: () => void;
  state: PlaybackState;
  error: string | null;
  onMessage: (callback: (msg: WSMessage) => void) => void;
}

export const useWebSocket = (url: string): UseWebSocketReturn => {
  const [state, setState] = useState<PlaybackState>('idle');
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const messageCallbackRef = useRef<((msg: WSMessage) => void) | null>(null);

  const connect = useCallback(async () => {
    return new Promise<void>((resolve, reject) => {
      setState('connecting');
      setError(null);

      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('✅ WebSocket connected');
        setState('idle');
        resolve();
      };

      ws.onmessage = (event) => {
        try {
          const msg: WSMessage = JSON.parse(event.data);
          messageCallbackRef.current?.(msg);

          if (msg.type === 'audio_chunk') {
            setState('playing');
          } else if (msg.type === 'end') {
            setState('idle');
          } else if (msg.type === 'error') {
            setState('error');
            setError(msg.message || 'Unknown error');
          }
        } catch (e) {
          console.error('Failed to parse message:', e);
        }
      };

      ws.onerror = (e) => {
        console.error('❌ WebSocket error:', e);
        setState('error');
        setError('Connection error');
        reject(e);
      };

      ws.onclose = () => {
        console.log('🔌 WebSocket closed');
        setState('idle');
        wsRef.current = null;
      };
    });
  }, [url]);

  const send = useCallback((request: TTSRequest) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(request));
    } else {
      console.error('WebSocket not connected');
    }
  }, []);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setState('idle');
  }, []);

  const onMessage = useCallback((callback: (msg: WSMessage) => void) => {
    messageCallbackRef.current = callback;
  }, []);

  return { connect, send, disconnect, state, error, onMessage };
};