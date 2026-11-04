import React, { useState, useRef, useEffect } from 'react';
import { Mic, Trash2, Send, Play, Pause, AlertCircle } from 'lucide-react';
import { formatDuration } from './staffChatUtils';

export function StaffVoiceRecorder({ onSendVoice, onCancel }) {
  const [isRecording, setIsRecording] = useState(false);
  const [duration, setDuration] = useState(0);
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [isPlayingPreview, setIsPlayingPreview] = useState(false);
  const [error, setError] = useState(null);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const animationFrameRef = useRef(null);
  const previewAudioRef = useRef(null);
  const [waveHeights, setWaveHeights] = useState([12, 18, 24, 16, 28, 14, 20, 26, 12, 22]);

  useEffect(() => {
    startRecording();
    return () => {
      cleanup();
    };
  }, []);

  const cleanup = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      try {
        mediaRecorderRef.current.stop();
        mediaRecorderRef.current.stream.getTracks().forEach((track) => track.stop());
      } catch (e) {}
    }
    if (audioContextRef.current) {
      try {
        audioContextRef.current.close();
      } catch (e) {}
    }
  };

  const startRecording = async () => {
    try {
      setError(null);
      audioChunksRef.current = [];
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // Audio analysis for real-time waveform animation
      try {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        const audioCtx = new AudioCtx();
        audioContextRef.current = audioCtx;
        const source = audioCtx.createMediaStreamSource(stream);
        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 32;
        source.connect(analyser);
        analyserRef.current = analyser;

        const updateWaveform = () => {
          if (!analyserRef.current) return;
          const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
          analyserRef.current.getByteFrequencyData(dataArray);

          const bars = [];
          for (let i = 0; i < 12; i++) {
            const val = dataArray[i] || 10;
            const h = Math.max(8, Math.min(32, Math.round((val / 255) * 32)));
            bars.push(h);
          }
          setWaveHeights(bars);
          animationFrameRef.current = requestAnimationFrame(updateWaveform);
        };
        updateWaveform();
      } catch (e) {
        // Fallback to random wave if AudioContext unavailable
      }

      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm;codecs=opus' });
        setAudioBlob(blob);
        const url = URL.createObjectURL(blob);
        setAudioUrl(url);
      };

      mediaRecorder.start(100);
      setIsRecording(true);
      setDuration(0);

      timerRef.current = setInterval(() => {
        setDuration((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      console.error('Microphone error:', err);
      setError('Microphone permission denied or device not found.');
    }
  };

  const stopAndPreview = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach((t) => t.stop());
    }
    setIsRecording(false);
  };

  const handleSend = () => {
    if (isRecording) {
      if (timerRef.current) clearInterval(timerRef.current);
      if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
        mediaRecorderRef.current.addEventListener('stop', () => {
          const blob = new Blob(audioChunksRef.current, { type: 'audio/webm;codecs=opus' });
          onSendVoice(blob, duration);
        }, { once: true });
        mediaRecorderRef.current.stop();
        mediaRecorderRef.current.stream.getTracks().forEach((t) => t.stop());
      }
    } else if (audioBlob) {
      onSendVoice(audioBlob, duration);
    }
  };

  const handleTogglePreviewPlay = () => {
    if (!previewAudioRef.current) return;
    if (isPlayingPreview) {
      previewAudioRef.current.pause();
      setIsPlayingPreview(false);
    } else {
      previewAudioRef.current.play();
      setIsPlayingPreview(true);
    }
  };

  if (error) {
    return (
      <div className="staff-voice-record-bar" style={{ background: 'rgba(239, 68, 68, 0.15)', border: '1px solid #ef4444' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#f87171', fontSize: 13 }}>
          <AlertCircle size={16} />
          <span>{error}</span>
        </div>
        <button type="button" className="staff-composer-btn" onClick={onCancel} title="Cancel">
          <Trash2 size={18} />
        </button>
      </div>
    );
  }

  return (
    <div className="staff-voice-record-bar">
      {/* Left cancel / delete button */}
      <button
        type="button"
        className="staff-composer-btn"
        onClick={() => {
          cleanup();
          onCancel();
        }}
        title="Cancel recording"
        style={{ color: '#ef4444' }}
      >
        <Trash2 size={18} />
      </button>

      {/* Middle: live wave & timer or preview controls */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, flex: 1, justifyContent: 'center' }}>
        {isRecording ? (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className="status-dot busy pulse" style={{ width: 10, height: 10 }} />
              <span style={{ fontFamily: 'monospace', fontSize: 13, fontWeight: 700, color: '#e9edef' }}>
                {formatDuration(duration)}
              </span>
            </div>

            {/* Live animated waveform bars */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 3, height: 32 }}>
              {waveHeights.map((h, i) => (
                <div
                  key={i}
                  style={{
                    width: 3,
                    height: `${h}px`,
                    background: 'var(--p)',
                    borderRadius: 2,
                    transition: 'height 0.08s ease',
                  }}
                />
              ))}
            </div>

            <button
              type="button"
              onClick={stopAndPreview}
              style={{
                fontSize: 11,
                padding: '3px 8px',
                borderRadius: 4,
                background: 'rgba(255,255,255,0.1)',
                color: '#94a3b8',
                border: 'none',
                cursor: 'pointer'
              }}
            >
              Preview
            </button>
          </>
        ) : (
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            {audioUrl && (
              <audio
                ref={previewAudioRef}
                src={audioUrl}
                onEnded={() => setIsPlayingPreview(false)}
              />
            )}
            <button
              type="button"
              onClick={handleTogglePreviewPlay}
              className="staff-composer-btn"
              style={{ color: 'var(--p)' }}
            >
              {isPlayingPreview ? <Pause size={18} /> : <Play size={18} />}
            </button>
            <span style={{ fontSize: 12, color: '#94a3b8', fontWeight: 600 }}>
              Preview Recording ({formatDuration(duration)})
            </span>
          </div>
        )}
      </div>

      {/* Right send button */}
      <button
        type="button"
        className="staff-composer-btn staff-composer-send-btn"
        onClick={handleSend}
        title="Send voice message"
      >
        <Send size={18} />
      </button>
    </div>
  );
}
