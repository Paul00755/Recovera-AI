import React, { useState, useRef, useEffect, useMemo } from 'react';
import { Play, Pause } from 'lucide-react';
import { formatDuration, API } from './staffChatUtils';

const SPEEDS = [1, 1.5, 2];

export function StaffVoicePlayer({ voiceUrl, duration = 0, isOutgoing = false }) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [speedIndex, setSpeedIndex] = useState(0); // 1x, 1.5x, 2x
  const [blobUrl, setBlobUrl] = useState('');
  const audioRef = useRef(null);

  const totalDuration = duration || 12;

  // Retrieve auth token for media streaming
  const authenticatedUrl = useMemo(() => {
    if (!voiceUrl) return '';
    const base = voiceUrl.startsWith('http') ? voiceUrl : `${API}${voiceUrl}`;
    try {
      const raw = sessionStorage.getItem('recovera_session') || localStorage.getItem('recovera_session');
      if (raw) {
        const parsed = JSON.parse(raw);
        if (parsed?.token && !base.includes('token=')) {
          const sep = base.includes('?') ? '&' : '?';
          return `${base}${sep}token=${encodeURIComponent(parsed.token)}`;
        }
      }
    } catch (e) {}
    return base;
  }, [voiceUrl]);

  // Static bars representing an audio waveform
  const waveformBars = [
    8, 14, 22, 16, 28, 20, 12, 18, 26, 30, 24, 18, 14, 22, 28, 20, 16, 24, 18, 12, 16, 22, 14, 10
  ];

  // Fallback blob fetcher if browser blocks direct streaming
  useEffect(() => {
    let active = true;
    let createdUrl = '';

    const tryFetchBlob = async () => {
      if (!voiceUrl) return;
      try {
        let headers = {};
        const raw = localStorage.getItem('recovera_session');
        if (raw) {
          const parsed = JSON.parse(raw);
          if (parsed?.token) headers.Authorization = `Bearer ${parsed.token}`;
        }
        const res = await fetch(authenticatedUrl || voiceUrl, { headers });
        if (res.ok) {
          const blob = await res.blob();
          if (active && blob.size > 0) {
            createdUrl = URL.createObjectURL(blob);
            setBlobUrl(createdUrl);
          }
        }
      } catch (err) {
        console.warn('Voice blob fallback error:', err);
      }
    };

    tryFetchBlob();

    return () => {
      active = false;
      if (createdUrl) URL.revokeObjectURL(createdUrl);
    };
  }, [voiceUrl, authenticatedUrl]);

  const effectiveSrc = blobUrl || authenticatedUrl || voiceUrl;

  useEffect(() => {
    if (audioRef.current && effectiveSrc) {
      try {
        audioRef.current.load();
      } catch (e) {}
    }
  }, [effectiveSrc]);

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play().then(() => {
        setIsPlaying(true);
      }).catch((e) => {
        console.error('Audio playback error:', e);
        // Retry with blobUrl if not already active
        if (blobUrl && audioRef.current.src !== blobUrl) {
          audioRef.current.src = blobUrl;
          audioRef.current.load();
          audioRef.current.play().then(() => setIsPlaying(true)).catch(() => {});
        }
      });
    }
  };

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
    }
  };

  const handleEnded = () => {
    setIsPlaying(false);
    setCurrentTime(0);
  };

  const cycleSpeed = () => {
    const nextIdx = (speedIndex + 1) % SPEEDS.length;
    setSpeedIndex(nextIdx);
    const speed = SPEEDS[nextIdx];
    if (audioRef.current) {
      audioRef.current.playbackRate = speed;
    }
  };

  const handleScrub = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const pct = Math.max(0, Math.min(1, clickX / rect.width));
    const newTime = pct * (audioRef.current?.duration || totalDuration);
    if (audioRef.current) {
      audioRef.current.currentTime = newTime;
    }
    setCurrentTime(newTime);
  };

  const progressPct = (currentTime / (audioRef.current?.duration || totalDuration)) * 100;

  return (
    <div
      className={`staff-voice-player-bubble ${isOutgoing ? 'outgoing' : 'incoming'}`}
      style={{ minWidth: 260, padding: '4px 2px' }}
    >
      <audio
        ref={audioRef}
        src={effectiveSrc}
        onTimeUpdate={handleTimeUpdate}
        onEnded={handleEnded}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onError={(err) => console.warn('Audio tag error:', err)}
        preload="auto"
      />

      <button
        type="button"
        onClick={togglePlay}
        className="staff-voice-play-btn"
        title={isPlaying ? 'Pause' : 'Play Voice Message'}
      >
        {isPlaying ? <Pause size={18} /> : <Play size={18} style={{ marginLeft: 2 }} />}
      </button>

      {/* Waveform Scrubber */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 5 }}>
        <div
          onClick={handleScrub}
          style={{
            height: 34,
            display: 'flex',
            alignItems: 'center',
            gap: 2.5,
            cursor: 'pointer',
            position: 'relative',
          }}
          title="Click to seek"
        >
          {waveformBars.map((height, i) => {
            const barPct = (i / waveformBars.length) * 100;
            const isPlayed = barPct <= progressPct;
            return (
              <div
                key={i}
                style={{
                  width: 3.5,
                  height: `${Math.max(8, height)}px`,
                  borderRadius: 3,
                  background: isPlayed
                    ? 'var(--voice-played, #00a884)'
                    : 'var(--voice-unplayed, #8696a0)',
                  transition: 'background 0.1s ease, transform 0.1s ease',
                  transform: isPlaying && isPlayed ? 'scaleY(1.08)' : 'scaleY(1)'
                }}
              />
            );
          })}
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{
            fontSize: 11,
            color: 'var(--voice-text, #54656f)',
            fontFamily: 'monospace',
            fontWeight: 600
          }}>
            {formatDuration(isPlaying ? currentTime : totalDuration)}
          </span>

          <button
            type="button"
            onClick={cycleSpeed}
            className="staff-voice-speed-pill"
            title="Toggle playback speed"
            style={{
              padding: '2px 8px',
              borderRadius: 999,
              fontSize: 11,
              fontWeight: 700,
              cursor: 'pointer',
              background: 'var(--voice-pill-bg, rgba(0,0,0,0.08))',
              color: 'var(--voice-pill-color, #111b21)',
              border: '1px solid var(--voice-pill-border, rgba(0,0,0,0.15))',
              transition: 'background 0.15s ease'
            }}
          >
            {SPEEDS[speedIndex]}x
          </button>
        </div>
      </div>
    </div>
  );
}
