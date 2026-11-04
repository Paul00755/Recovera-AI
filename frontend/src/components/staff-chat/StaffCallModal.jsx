import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { Phone, PhoneOff, Video, VideoOff, Mic, MicOff, Volume2, Maximize2, Minimize2 } from 'lucide-react';
import { formatDuration, RTC_CONFIG } from './staffChatUtils';
import { StaffAvatar } from './StaffAvatar';

// High-fidelity Web Audio Ringtone Synthesizer with immediate audio cancellation
function startRingtoneChime() {
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return () => {};
    const ctx = new AudioCtx();
    let isPlaying = true;
    let activeOscs = [];

    const stopAllOscs = () => {
      activeOscs.forEach(o => {
        try { o.stop(0); o.disconnect(); } catch (e) {}
      });
      activeOscs = [];
    };

    const playTone = () => {
      if (!isPlaying || ctx.state === 'closed') return;
      if (ctx.state === 'suspended') ctx.resume();

      const now = ctx.currentTime;
      const osc1 = ctx.createOscillator();
      const osc2 = ctx.createOscillator();
      const gain = ctx.createGain();

      osc1.type = 'sine';
      osc2.type = 'sine';
      osc1.frequency.setValueAtTime(440, now);
      osc2.frequency.setValueAtTime(480, now);

      gain.gain.setValueAtTime(0.05, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 1.1);

      osc1.connect(gain);
      osc2.connect(gain);
      gain.connect(ctx.destination);

      activeOscs.push(osc1, osc2);

      osc1.start(now);
      osc2.start(now);
      osc1.stop(now + 1.1);
      osc2.stop(now + 1.1);

      setTimeout(() => {
        activeOscs = activeOscs.filter(o => o !== osc1 && o !== osc2);
      }, 1200);
    };

    playTone();
    const interval = setInterval(playTone, 2200);

    return () => {
      isPlaying = false;
      clearInterval(interval);
      stopAllOscs();
      try { ctx.close(); } catch (e) {}
    };
  } catch (e) {
    return () => {};
  }
}

// Memoized Video Viewport: isolated from duration timer ticks to eliminate all video flickering
const VideoViewport = React.memo(({ isPip, remoteVideoRef, localVideoRef }) => {
  return (
    <div style={{
      position: 'relative',
      width: '100%',
      height: isPip ? 180 : 380,
      background: '#000',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      overflow: 'hidden',
      transform: 'translateZ(0)',
      willChange: 'transform'
    }}>
      <video
        ref={remoteVideoRef}
        autoPlay
        playsInline
        style={{
          width: '100%',
          height: '100%',
          objectFit: 'cover',
          background: '#000',
          transform: 'translateZ(0)',
          willChange: 'transform'
        }}
      />

      {/* Self Video PIP Preview */}
      <div
        style={{
          position: 'absolute',
          bottom: 10,
          right: 10,
          width: isPip ? 65 : 120,
          aspectRatio: '4 / 3',
          borderRadius: 6,
          overflow: 'hidden',
          border: '1.5px solid var(--p)',
          boxShadow: '0 4px 14px rgba(0,0,0,0.6)',
          background: '#111b21',
          zIndex: 2,
          transform: 'translateZ(0)',
          willChange: 'transform'
        }}
      >
        <video
          ref={localVideoRef}
          autoPlay
          playsInline
          muted
          style={{ width: '100%', height: '100%', objectFit: 'cover', transform: 'scaleX(-1) translateZ(0)', willChange: 'transform' }}
        />
      </div>
    </div>
  );
});

export function StaffCallModal({
  callSession, // { callId, isCaller, callType: 'voice' | 'video', peerUser }
  wsRef,
  onEndCall,
}) {
  const { callId, isCaller, callType, peerUser } = callSession;
  const isVideo = callType === 'video';

  const [callStatus, setCallStatus] = useState(isCaller ? 'Ringing...' : 'Connecting...');
  const [duration, setDuration] = useState(0);
  const [isMuted, setIsMuted] = useState(false);
  const [isVideoOff, setIsVideoOff] = useState(false);
  const [isPip, setIsPip] = useState(false);

  // Draggable PIP coordinates
  const [pipPos, setPipPos] = useState(() => ({
    x: typeof window !== 'undefined' ? window.innerWidth - 350 : 800,
    y: typeof window !== 'undefined' ? window.innerHeight - 300 : 500,
  }));
  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef({ mouseX: 0, mouseY: 0, startX: 0, startY: 0 });

  const localVideoRef = useRef(null);
  const remoteVideoRef = useRef(null);
  const remoteAudioRef = useRef(null);
  const peerConnectionRef = useRef(null);
  const localStreamRef = useRef(null);
  const remoteStreamRef = useRef(null);
  const timerRef = useRef(null);
  const stopRingtoneRef = useRef(null);
  const pendingCandidatesRef = useRef([]);

  const stopActiveRingtone = useCallback(() => {
    if (stopRingtoneRef.current) {
      stopRingtoneRef.current();
      stopRingtoneRef.current = null;
    }
    try {
      window.dispatchEvent(new CustomEvent('staff_call_stop_ringtone'));
    } catch (e) {}
  }, []);

  const sendWs = useCallback((payload) => {
    if (wsRef?.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(payload));
    }
  }, [wsRef]);

  const cleanupCall = useCallback(() => {
    stopActiveRingtone();
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach((track) => track.stop());
      localStreamRef.current = null;
    }
    if (peerConnectionRef.current) {
      try {
        peerConnectionRef.current.close();
      } catch (e) {}
      peerConnectionRef.current = null;
    }
  }, [stopActiveRingtone]);

  const handleHangup = useCallback(() => {
    cleanupCall();
    sendWs({
      type: 'call_hangup',
      target_user_id: peerUser?.id,
      call_id: callId,
    });
    onEndCall(duration);
  }, [cleanupCall, sendWs, peerUser?.id, callId, onEndCall, duration]);

  // Persistent track re-attachment effect across PIP state flips
  useEffect(() => {
    if (isVideo) {
      if (remoteVideoRef.current && remoteStreamRef.current && remoteVideoRef.current.srcObject !== remoteStreamRef.current) {
        remoteVideoRef.current.srcObject = remoteStreamRef.current;
      }
      if (localVideoRef.current && localStreamRef.current && localVideoRef.current.srcObject !== localStreamRef.current) {
        localVideoRef.current.srcObject = localStreamRef.current;
      }
    }
  }, [isVideo, isPip]);

  // Draggable PIP window with mouse + touch support
  const handleDragStart = (e) => {
    if (!isPip) return;
    if (e.target.closest('button')) return;
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    setIsDragging(true);
    dragStartRef.current = {
      mouseX: clientX,
      mouseY: clientY,
      startX: pipPos.x,
      startY: pipPos.y,
    };
  };

  useEffect(() => {
    if (!isDragging) return;
    const handleMove = (e) => {
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      const clientY = e.touches ? e.touches[0].clientY : e.clientY;
      const dx = clientX - dragStartRef.current.mouseX;
      const dy = clientY - dragStartRef.current.mouseY;
      const newX = Math.max(8, Math.min(window.innerWidth - 330, dragStartRef.current.startX + dx));
      const newY = Math.max(8, Math.min(window.innerHeight - 240, dragStartRef.current.startY + dy));
      setPipPos({ x: newX, y: newY });
    };
    const handleEnd = () => setIsDragging(false);

    window.addEventListener('mousemove', handleMove);
    window.addEventListener('mouseup', handleEnd);
    window.addEventListener('touchmove', handleMove, { passive: true });
    window.addEventListener('touchend', handleEnd);
    return () => {
      window.removeEventListener('mousemove', handleMove);
      window.removeEventListener('mouseup', handleEnd);
      window.removeEventListener('touchmove', handleMove);
      window.removeEventListener('touchend', handleEnd);
    };
  }, [isDragging]);

  // WebRTC initialization and connection management
  useEffect(() => {
    let isCancelled = false;

    // Start caller ringtone if caller is waiting for answer
    if (isCaller) {
      stopRingtoneRef.current = startRingtoneChime();
    }

    const initWebRTC = async () => {
      try {
        const pc = new RTCPeerConnection(RTC_CONFIG);
        peerConnectionRef.current = pc;

        // ICE candidate handler
        pc.onicecandidate = (event) => {
          if (event.candidate && peerUser?.id) {
            sendWs({
              type: 'webrtc_ice_candidate',
              target_user_id: peerUser.id,
              candidate: event.candidate,
              signal: event.candidate,
              call_id: callId,
            });
          }
        };

        // Remote track arrival (call answered & media streaming)
        pc.ontrack = (event) => {
          if (isCancelled) return;
          stopActiveRingtone();
          setCallStatus('Connected');
          if (!timerRef.current) {
            timerRef.current = setInterval(() => setDuration((d) => d + 1), 1000);
          }

          const stream = event.streams[0];
          remoteStreamRef.current = stream;
          if (isVideo && remoteVideoRef.current) {
            if (remoteVideoRef.current.srcObject !== stream) {
              remoteVideoRef.current.srcObject = stream;
            }
          } else if (remoteAudioRef.current) {
            if (remoteAudioRef.current.srcObject !== stream) {
              remoteAudioRef.current.srcObject = stream;
            }
          }
        };

        // Local media capture
        const constraints = {
          audio: true,
          video: isVideo ? { width: { ideal: 1280 }, height: { ideal: 720 } } : false,
        };

        let stream = null;
        try {
          stream = await navigator.mediaDevices.getUserMedia(constraints);
        } catch (mediaErr) {
          console.warn('getUserMedia fallback to audio only:', mediaErr);
          stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        }

        if (isCancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }

        localStreamRef.current = stream;
        if (isVideo && localVideoRef.current) {
          localVideoRef.current.srcObject = stream;
        }

        stream.getTracks().forEach((track) => pc.addTrack(track, stream));

        // If caller, send WebRTC Offer
        if (isCaller) {
          const offer = await pc.createOffer();
          await pc.setLocalDescription(offer);
          sendWs({
            type: 'webrtc_offer',
            target_user_id: peerUser.id,
            signal: offer,
            call_id: callId,
          });
        }
      } catch (err) {
        console.error('WebRTC initialization failed:', err);
        setCallStatus('Call Failed');
        stopActiveRingtone();
      }
    };

    initWebRTC();

    // Signal listener for WebSocket message dispatch
    const handleSignalEvent = async (e) => {
      const { type, data } = e.detail || {};
      const pc = peerConnectionRef.current;
      if (!pc) return;

      if (type === 'call_accept') {
        stopActiveRingtone();
        setCallStatus('Connected');
        if (!timerRef.current) {
          timerRef.current = setInterval(() => setDuration((d) => d + 1), 1000);
        }
      } else if (type === 'webrtc_offer' && data?.signal) {
        try {
          stopActiveRingtone();
          await pc.setRemoteDescription(new RTCSessionDescription(data.signal));
          while (pendingCandidatesRef.current.length > 0) {
            const cand = pendingCandidatesRef.current.shift();
            try { await pc.addIceCandidate(new RTCIceCandidate(cand)); } catch (e) {}
          }
          const answer = await pc.createAnswer();
          await pc.setLocalDescription(answer);
          sendWs({
            type: 'webrtc_answer',
            target_user_id: peerUser.id,
            signal: answer,
            call_id: callId,
          });
          setCallStatus('Connected');
          if (!timerRef.current) {
            timerRef.current = setInterval(() => setDuration((d) => d + 1), 1000);
          }
        } catch (err) {
          console.error('Error handling webrtc_offer:', err);
        }
      } else if (type === 'webrtc_answer' && data?.signal) {
        try {
          stopActiveRingtone();
          await pc.setRemoteDescription(new RTCSessionDescription(data.signal));
          while (pendingCandidatesRef.current.length > 0) {
            const cand = pendingCandidatesRef.current.shift();
            try { await pc.addIceCandidate(new RTCIceCandidate(cand)); } catch (e) {}
          }
          setCallStatus('Connected');
          if (!timerRef.current) {
            timerRef.current = setInterval(() => setDuration((d) => d + 1), 1000);
          }
        } catch (err) {
          console.error('Error handling webrtc_answer:', err);
        }
      } else if (type === 'webrtc_ice_candidate' && data?.signal) {
        try {
          if (pc.remoteDescription) {
            await pc.addIceCandidate(new RTCIceCandidate(data.signal));
          } else {
            pendingCandidatesRef.current.push(data.signal);
          }
        } catch (err) {}
      } else if (type === 'call_reject') {
        stopActiveRingtone();
        setCallStatus('Call Declined');
        setTimeout(() => handleHangup(), 2000);
      } else if (type === 'call_hangup') {
        stopActiveRingtone();
        setCallStatus('Call Ended');
        setTimeout(() => handleHangup(), 1500);
      }
    };

    window.addEventListener('staff_call_signal', handleSignalEvent);

    return () => {
      isCancelled = true;
      window.removeEventListener('staff_call_signal', handleSignalEvent);
      cleanupCall();
    };
  }, [callId, isCaller, isVideo, peerUser?.id, sendWs, cleanupCall, handleHangup, stopActiveRingtone]);

  const handleToggleMute = () => {
    if (localStreamRef.current) {
      const audioTrack = localStreamRef.current.getAudioTracks()[0];
      if (audioTrack) {
        audioTrack.enabled = !audioTrack.enabled;
        setIsMuted(!audioTrack.enabled);
      }
    }
  };

  const handleToggleVideo = () => {
    if (localStreamRef.current) {
      const videoTrack = localStreamRef.current.getVideoTracks()[0];
      if (videoTrack) {
        videoTrack.enabled = !videoTrack.enabled;
        setIsVideoOff(!videoTrack.enabled);
      }
    }
  };

  // Picture-in-Picture draggable floating container style
  const pipContainerStyle = isPip ? {
    position: 'fixed',
    left: `${pipPos.x}px`,
    top: `${pipPos.y}px`,
    width: 320,
    zIndex: 99999,
    background: '#111b21',
    borderRadius: 14,
    boxShadow: '0 16px 45px rgba(0, 0, 0, 0.8)',
    border: '1.5px solid var(--p)',
    overflow: 'hidden',
    userSelect: 'none',
    cursor: isDragging ? 'grabbing' : 'default',
  } : {};

  return (
    <div
      className={isPip ? '' : 'staff-call-modal-overlay'}
      style={isPip ? { position: 'fixed', inset: 0, pointerEvents: 'none', zIndex: 99998 } : {}}
    >
      <div
        className={isPip ? '' : 'staff-call-box'}
        style={{ ...pipContainerStyle, pointerEvents: 'auto' }}
        onMouseDown={handleDragStart}
        onTouchStart={handleDragStart}
      >
        {/* Call Header */}
        <div style={{
          padding: isPip ? '8px 12px' : '14px 20px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          borderBottom: '1px solid rgba(255,255,255,0.08)',
          background: '#0d141a',
          cursor: isPip ? 'move' : 'default'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, overflow: 'hidden' }}>
            <span className={`status-dot ${callStatus === 'Connected' ? 'online' : 'away pulse'}`} />
            <span style={{ fontSize: isPip ? 12 : 13, fontWeight: 700, color: '#e2e8f0', whiteSpace: 'nowrap', textOverflow: 'ellipsis', overflow: 'hidden' }}>
              {peerUser?.name} {isPip ? '' : `• ${isVideo ? 'Video' : 'Voice'} Call`}
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ fontFamily: 'monospace', fontSize: 12, fontWeight: 700, color: 'var(--p)' }}>
              {callStatus === 'Connected' ? formatDuration(duration) : callStatus}
            </span>

            {/* Native OS Picture-in-Picture Button */}
            {isVideo && typeof document !== 'undefined' && document.pictureInPictureEnabled && (
              <button
                type="button"
                onClick={async () => {
                  try {
                    if (document.pictureInPictureElement) {
                      await document.exitPictureInPicture();
                    } else if (remoteVideoRef.current) {
                      await remoteVideoRef.current.requestPictureInPicture();
                    }
                  } catch (e) {
                    setIsPip((prev) => !prev);
                  }
                }}
                className="btn btn-ghost btn-xs"
                style={{
                  color: '#fff',
                  padding: '3px 6px',
                  borderRadius: 6,
                  background: 'rgba(255,255,255,0.1)'
                }}
                title="OS Pop-out Window (Picture-in-Picture across all applications)"
              >
                <Video size={13} />
              </button>
            )}

            {/* In-app Picture-in-Picture Toggle Button */}
            <button
              type="button"
              onClick={() => setIsPip((prev) => !prev)}
              className="btn btn-ghost btn-xs"
              style={{
                color: '#fff',
                padding: '3px 6px',
                borderRadius: 6,
                background: 'rgba(255,255,255,0.1)'
              }}
              title={isPip ? 'Expand Call View' : 'Picture-in-Picture (Drag & work while calling)'}
            >
              {isPip ? <Maximize2 size={14} /> : <Minimize2 size={14} />}
            </button>
          </div>
        </div>

        {/* Call Viewport: Memoized to eliminate flickering during timer updates */}
        {isVideo ? (
          <VideoViewport
            isPip={isPip}
            remoteVideoRef={remoteVideoRef}
            localVideoRef={localVideoRef}
          />
        ) : (
          <div style={{
            padding: isPip ? '24px 16px' : '40px 24px',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 12,
            background: '#0c131d'
          }}>
            <audio ref={remoteAudioRef} autoPlay />
            <StaffAvatar
              name={peerUser?.name}
              avatarUrl={peerUser?.avatar_url}
              status={callStatus === 'Connected' ? 'online' : 'away'}
              size={isPip ? 56 : 84}
              fontSize={isPip ? 20 : 28}
            />

            <div style={{ textAlign: 'center' }}>
              <h3 style={{ fontSize: isPip ? 15 : 18, fontWeight: 800, color: '#fff', marginBottom: 2 }}>
                {peerUser?.name}
              </h3>
              <p style={{ fontSize: 11.5, color: 'var(--p)', fontWeight: 600 }}>
                {peerUser?.specialization || peerUser?.role}
              </p>
              <p style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>
                {callStatus}
              </p>
            </div>
          </div>
        )}

        {/* Call Controls Bar */}
        <div style={{
          padding: isPip ? '10px 14px' : '14px 20px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: isPip ? 12 : 16,
          background: '#111b21',
          borderTop: '1px solid rgba(255,255,255,0.08)'
        }}>
          <button
            type="button"
            className={`webrtc-ctrl-btn ${isMuted ? 'active-off' : ''}`}
            onClick={handleToggleMute}
            style={{ width: isPip ? 36 : 44, height: isPip ? 36 : 44 }}
            title={isMuted ? 'Unmute' : 'Mute'}
          >
            {isMuted ? <MicOff size={isPip ? 16 : 19} /> : <Mic size={isPip ? 16 : 19} />}
          </button>

          {isVideo && (
            <button
              type="button"
              className={`webrtc-ctrl-btn ${isVideoOff ? 'active-off' : ''}`}
              onClick={handleToggleVideo}
              style={{ width: isPip ? 36 : 44, height: isPip ? 36 : 44 }}
              title={isVideoOff ? 'Turn video on' : 'Turn video off'}
            >
              {isVideoOff ? <VideoOff size={isPip ? 16 : 19} /> : <Video size={isPip ? 16 : 19} />}
            </button>
          )}

          <button
            type="button"
            className="webrtc-ctrl-btn btn-hangup"
            onClick={handleHangup}
            title="End Call"
            style={{ width: isPip ? 40 : 48, height: isPip ? 40 : 48, background: '#ef4444' }}
          >
            <PhoneOff size={isPip ? 18 : 22} />
          </button>
        </div>
      </div>
    </div>
  );
}
