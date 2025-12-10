// src/types/index.ts

export interface Voice {
    id: string;
    name: string;
    audioPath: string;
    textPath: string;
    region: 'north' | 'south';
    gender: 'male' | 'female';
  }
  
export interface TTSConfig {
backbone: string;
codec: string;
device: 'auto' | 'cpu' | 'cuda';
}
  
export interface WSMessage {
type: 'metadata' | 'audio_chunk' | 'end' | 'error' | 'status';
data?: string; // base64 audio
sample_rate?: number;
channels?: number;
length?: number;
message?: string;
}
  
export interface TTSRequest {
text: string;
voice: string;
mode: 'standard' | 'streaming';
config?: TTSConfig;
}

export type PlaybackState = 'idle' | 'connecting' | 'playing' | 'paused' | 'error';

export interface AudioChunk {
data: Float32Array;
timestamp: number;
}