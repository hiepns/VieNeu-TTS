// src/hooks/useAudioPlayer.ts

import { useState, useRef, useCallback, useEffect } from 'react';
import { AudioChunk } from '../types';

interface UseAudioPlayerReturn {
  initialize: (sampleRate: number) => void;
  enqueue: (audioData: Float32Array) => void;
  stop: () => void;
  isPlaying: boolean;
  analyser: AnalyserNode | null;
}

export const useAudioPlayer = (): UseAudioPlayerReturn => {
  const [isPlaying, setIsPlaying] = useState(false);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const queueRef = useRef<AudioChunk[]>([]);
  const playingRef = useRef(false);

  const initialize = useCallback((sampleRate: number) => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate,
      });

      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 2048;
      analyserRef.current.connect(audioContextRef.current.destination);
    }
  }, []);

  const playQueue = useCallback(async () => {
    if (!audioContextRef.current || playingRef.current) return;
    
    playingRef.current = true;
    setIsPlaying(true);

    while (queueRef.current.length > 0) {
      const chunk = queueRef.current.shift();
      if (!chunk) continue;

      const buffer = audioContextRef.current.createBuffer(
        1,
        chunk.data.length,
        audioContextRef.current.sampleRate
      );
      buffer.getChannelData(0).set(chunk.data);

      const source = audioContextRef.current.createBufferSource();
      source.buffer = buffer;
      source.connect(analyserRef.current!);

      await new Promise<void>((resolve) => {
        source.onended = () => resolve();
        source.start();
      });
    }

    playingRef.current = false;
    setIsPlaying(false);
  }, []);

  const enqueue = useCallback((audioData: Float32Array) => {
    queueRef.current.push({
      data: audioData,
      timestamp: Date.now(),
    });

    if (!playingRef.current) {
      playQueue();
    }
  }, [playQueue]);

  const stop = useCallback(() => {
    queueRef.current = [];
    playingRef.current = false;
    setIsPlaying(false);

    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
      analyserRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => {
      stop();
    };
  }, [stop]);

  return {
    initialize,
    enqueue,
    stop,
    isPlaying,
    analyser: analyserRef.current,
  };
};