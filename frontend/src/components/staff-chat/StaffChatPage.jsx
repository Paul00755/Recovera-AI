import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import {
  Search, Plus, Phone, PhoneOff, Video, MoreVertical, Send, Mic, Smile,
  Check, CheckCheck, X, Reply, Copy, Edit2, Trash2, ChevronLeft,
  ChevronRight, Volume2, Shield, AlertCircle, Ban, BellOff, Bell
} from 'lucide-react';
import { formatMessageDateSeparator, formatMessageTime, formatDuration, API, getWsBaseUrl } from './staffChatUtils';
import { StaffVoiceRecorder } from './StaffVoiceRecorder';
import { StaffVoicePlayer } from './StaffVoicePlayer';
import { StaffNewChatModal } from './StaffNewChatModal';
import { StaffContactInfoDrawer } from './StaffContactInfoDrawer';
import { StaffSearchDrawer } from './StaffSearchDrawer';
import { StaffCallModal } from './StaffCallModal';
import { StaffProfileModal } from './StaffProfileModal';
import { StaffAvatar } from './StaffAvatar';

const QUICK_EMOJIS = ['😊', '👍', '❤️', '🙏', '🩺', '🏥', '✌️', '😅', '👏', '🔥', '🎉', '💊'];

export function StaffChatPage({ session, navigate }) {
  const [conversations, setConversations] = useState([]);
  const [activeConvId, setActiveConvId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [staffDirectory, setStaffDirectory] = useState([]);
  const [filterTab, setFilterTab] = useState('all'); // all, unread, doctors, nurses, admins
  const [searchQuery, setSearchQuery] = useState('');
  const [inputText, setInputText] = useState('');
  const [replyingTo, setReplyingTo] = useState(null);
  const [editingMsg, setEditingMsg] = useState(null);
  const [activeContextMenuMsgId, setActiveContextMenuMsgId] = useState(null);

  // Panels & Modals
  const [isVoiceRecording, setIsVoiceRecording] = useState(false);
  const [emojiPickerOpen, setEmojiPickerOpen] = useState(false);
  const [searchDrawerOpen, setSearchDrawerOpen] = useState(false);
  const [contactDrawerOpen, setContactDrawerOpen] = useState(false);
  const [newChatModalOpen, setNewChatModalOpen] = useState(false);
  const [profileModalOpen, setProfileModalOpen] = useState(false);

  // Typing & Call states
  const [typingMap, setTypingMap] = useState({}); // convId -> typingUserName
  const [currentCallSession, setCurrentCallSession] = useState(null);
  const [incomingCallPrompt, setIncomingCallPrompt] = useState(null);
  const [offlineCallAlert, setOfflineCallAlert] = useState('');
  const ringtoneCleanupRef = useRef(null);

  const startIncomingSound = () => {
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      const ctx = new AudioCtx();
      let isPlaying = true;
      const playTone = () => {
        if (!isPlaying || ctx.state === 'closed') return;
        if (ctx.state === 'suspended') ctx.resume();
        const now = ctx.currentTime;
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(523.25, now);
        osc.frequency.exponentialRampToValueAtTime(659.25, now + 0.3);
        gain.gain.setValueAtTime(0.08, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.8);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start(now);
        osc.stop(now + 0.8);
      };
      playTone();
      const interval = setInterval(playTone, 1600);
      ringtoneCleanupRef.current = () => {
        isPlaying = false;
        clearInterval(interval);
        try { ctx.close(); } catch (e) {}
      };
    } catch (e) {}
  };

  const stopIncomingSound = () => {
    if (ringtoneCleanupRef.current) {
      ringtoneCleanupRef.current();
      ringtoneCleanupRef.current = null;
    }
  };

  const currentUserId = session?.user?.id;
  const token = session?.token || JSON.parse(sessionStorage.getItem('recovera_session') || localStorage.getItem('recovera_session') || '{}')?.token;

  const messagesEndRef = useRef(null);
  const chatBodyRef = useRef(null);
  const wsRef = useRef(null);
  const typingTimeoutRef = useRef(null);
  const activeConvIdRef = useRef(activeConvId);

  useEffect(() => {
    activeConvIdRef.current = activeConvId;
  }, [activeConvId]);

  // 1. Initial Load of Directory & Conversations
  useEffect(() => {
    loadDirectory();
    loadConversations();
    const handleDismissMenu = () => setActiveContextMenuMsgId(null);
    const handleStopSound = () => stopIncomingSound();
    window.addEventListener('click', handleDismissMenu);
    window.addEventListener('staff_call_stop_ringtone', handleStopSound);
    return () => {
      window.removeEventListener('click', handleDismissMenu);
      window.removeEventListener('staff_call_stop_ringtone', handleStopSound);
      stopIncomingSound();
    };
  }, []);

  const loadDirectory = async () => {
    try {
      const res = await fetch(`${API}/api/staff-chat/directory`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setStaffDirectory(data);
      }
    } catch (e) {
      console.error('Failed to load staff directory:', e);
    }
  };

  const loadConversations = async () => {
    try {
      const res = await fetch(`${API}/api/staff-chat/conversations`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setConversations(data);
        if (data.length > 0 && !activeConvId) {
          setActiveConvId(data[0].id);
        }
      }
    } catch (e) {
      console.error('Failed to load conversations:', e);
    }
  };

  // 2. Load Messages when Active Conversation changes
  useEffect(() => {
    if (!activeConvId) return;
    loadMessages(activeConvId);
    markConversationAsRead(activeConvId);
  }, [activeConvId]);

  const loadMessages = async (convId) => {
    try {
      const res = await fetch(`${API}/api/staff-chat/conversations/${convId}/messages`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setMessages(data);
        scrollToBottom();
      }
    } catch (e) {
      console.error('Failed to load messages:', e);
    }
  };

  const markConversationAsRead = async (convId) => {
    try {
      await fetch(`${API}/api/staff-chat/conversations/${convId}/read`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` }
      });
      setConversations((prev) =>
        prev.map((c) => (c.id === convId ? { ...c, unread_count: 0 } : c))
      );
    } catch (e) {}
  };

  // 3. WebSocket Real-time Connection
  useEffect(() => {
    if (!token) return;

    const wsUrl = `${getWsBaseUrl()}/ws/staff-chat?token=${token}`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      // console.log('Staff Chat WS connected');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleWsEvent(data);
      } catch (err) {
        console.error('WS Parse Error:', err);
      }
    };

    ws.onclose = () => {
      // console.log('Staff Chat WS disconnected');
    };

    return () => {
      ws.close();
    };
  }, [token, activeConvId]);

  const handleWsEvent = (event) => {
    const { type, conversation_id, message, data } = event;

    if (type === 'new_message' && message) {
      if (message.conversation_id === activeConvIdRef.current || message.conversation_id === activeConvId) {
        setMessages((prev) => {
          if (prev.some((m) => m.id === message.id)) return prev;
          return [...prev, message];
        });
        scrollToBottom();
        markConversationAsRead(message.conversation_id);
      }
      // Update preview in conversation list
      setConversations((prev) =>
        prev.map((c) => {
          if (c.id === message.conversation_id) {
            return {
              ...c,
              last_message: {
                content: message.content,
                message_type: message.message_type,
                created_at: message.created_at,
                sender_id: message.sender_id,
              },
              unread_count: message.conversation_id === activeConvId ? 0 : (c.unread_count || 0) + 1,
            };
          }
          return c;
        })
      );
    } else if (type === 'message_edited' && message) {
      setMessages((prev) => prev.map((m) => (m.id === message.id ? message : m)));
    } else if (type === 'message_deleted_for_everyone' && message) {
      setMessages((prev) => prev.map((m) => (m.id === message.id ? message : m)));
    } else if (type === 'typing_start') {
      setTypingMap((prev) => ({ ...prev, [conversation_id]: event.user_name }));
    } else if (type === 'typing_stop') {
      setTypingMap((prev) => {
        const next = { ...prev };
        delete next[conversation_id];
        return next;
      });
    } else if (type === 'presence_update' && data) {
      setStaffDirectory((prev) =>
        prev.map((s) => (s.id === data.user_id ? { ...s, availability_status: data.status } : s))
      );
      setConversations((prev) =>
        prev.map((c) => {
          const p = c.other_participant || c.other_user;
          if (p?.id === data.user_id) {
            const updated = { ...p, availability_status: data.status };
            return {
              ...c,
              other_participant: updated,
              other_user: updated
            };
          }
          return c;
        })
      );
    } else if (['webrtc_offer', 'webrtc_answer', 'webrtc_ice_candidate', 'call_accept', 'call_reject', 'call_hangup'].includes(type)) {
      window.dispatchEvent(new CustomEvent('staff_call_signal', { detail: { type, data: event } }));
      if (type === 'call_hangup' || type === 'call_reject') {
        stopIncomingSound();
        setIncomingCallPrompt(null);
      }
    } else if ((type === 'incoming_call' || type === 'call_invite') && (data || event)) {
      const callData = data || event;
      if (currentCallSession) {
        // Line busy in another call
        wsRef.current?.send(JSON.stringify({
          type: 'call_busy',
          target_user_id: callData.caller_id || callData.caller?.id,
          call_id: callData.call_id
        }));
        return;
      }
      const callerObj = callData.caller || {
        id: callData.caller_id,
        name: callData.caller_name,
        role: callData.caller_role,
        avatar_url: callData.caller_avatar,
        specialization: callData.caller_specialization
      };
      setIncomingCallPrompt({
        callId: callData.call_id,
        callType: callData.call_type || 'voice',
        caller: callerObj
      });
      startIncomingSound();
    }
  };

  const scrollToBottom = () => {
    setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, 100);
  };

  // Typing emitter
  const handleInputChange = (e) => {
    const val = e.target.value;
    setInputText(val);

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN && activeConvId) {
      wsRef.current.send(
        JSON.stringify({ type: 'typing_start', conversation_id: activeConvId })
      );

      if (typingTimeoutRef.current) clearTimeout(typingTimeoutRef.current);
      typingTimeoutRef.current = setTimeout(() => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(
            JSON.stringify({ type: 'typing_stop', conversation_id: activeConvId })
          );
        }
      }, 2000);
    }
  };

  // 4. Send Message (Text)
  const handleSendMessage = async (e) => {
    e?.preventDefault();
    if (!inputText.trim() || !activeConvId) return;

    const content = inputText.trim();
    setInputText('');
    const replyId = replyingTo?.id || null;
    setReplyingTo(null);

    if (editingMsg) {
      // Edit existing message
      try {
        const res = await fetch(`${API}/api/staff-chat/messages/${editingMsg.id}`, {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`
          },
          body: JSON.stringify({ content })
        });
        if (res.ok) {
          const updated = await res.json();
          setMessages((prev) => prev.map((m) => (m.id === updated.id ? updated : m)));
        }
      } catch (err) {
        console.error('Edit error:', err);
      } finally {
        setEditingMsg(null);
      }
      return;
    }

    try {
      const res = await fetch(`${API}/api/staff-chat/conversations/${activeConvId}/messages`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({
          message_type: 'text',
          content: content,
          reply_to_id: replyId,
        })
      });
      if (res.ok) {
        const newMsg = await res.json();
        setMessages((prev) => {
          if (prev.some((m) => m.id === newMsg.id)) return prev;
          return [...prev, newMsg];
        });
        scrollToBottom();
      }
    } catch (err) {
      console.error('Send error:', err);
    }
  };

  // 5. Send Voice Message
  const handleSendVoice = async (audioBlob, durationSeconds) => {
    if (!activeConvId) return;
    setIsVoiceRecording(false);

    try {
      const formData = new FormData();
      formData.append('file', audioBlob, `voice_${Date.now()}.webm`);

      const uploadRes = await fetch(`${API}/api/staff-chat/voice`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      });

      if (!uploadRes.ok) throw new Error('Voice upload failed');
      const { voice_url } = await uploadRes.json();

      const res = await fetch(`${API}/api/staff-chat/conversations/${activeConvId}/messages`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({
          message_type: 'voice',
          voice_url: voice_url,
          voice_duration: durationSeconds || 5.0,
          content: 'Voice Message',
          reply_to_id: replyingTo?.id || null,
        })
      });

      if (res.ok) {
        const newMsg = await res.json();
        setMessages((prev) => {
          if (prev.some((m) => m.id === newMsg.id)) return prev;
          return [...prev, newMsg];
        });
        scrollToBottom();
      }
    } catch (err) {
      console.error('Voice send error:', err);
    }
  };

  // 6. Delete Message (Me vs Everyone)
  const handleDeleteForMe = async (msgId) => {
    setActiveContextMenuMsgId(null);
    try {
      const res = await fetch(`${API}/api/staff-chat/messages/${msgId}/delete-for-me`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` }
      });
      if (res.ok) {
        setMessages((prev) => prev.filter((m) => m.id !== msgId));
      }
    } catch (err) {
      console.error('Delete for me error:', err);
    }
  };

  const handleDeleteForEveryone = async (msgId) => {
    setActiveContextMenuMsgId(null);
    if (!window.confirm('Delete this message for everyone in the conversation?')) return;
    try {
      const res = await fetch(`${API}/api/staff-chat/messages/${msgId}/delete-for-everyone`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` }
      });
      if (res.ok) {
        const updated = await res.json();
        setMessages((prev) => prev.map((m) => (m.id === updated.id ? updated : m)));
      }
    } catch (err) {
      console.error('Delete for everyone error:', err);
    }
  };

  // Copy text to clipboard
  const handleCopyText = (text) => {
    navigator.clipboard.writeText(text);
    setActiveContextMenuMsgId(null);
  };

  // Initiate call
  const handleInitiateCall = async (callType) => {
    const activeConv = conversations.find((c) => c.id === activeConvId);
    const peer = activeConv?.other_participant || activeConv?.other_user;
    if (!activeConv || !peer) return;

    if (peer.availability_status === 'offline') {
      fetch(`${API}/api/staff-chat/calls`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({
          receiver_id: peer.id,
          call_type: callType,
        })
      }).catch(() => {});
      setOfflineCallAlert(`${peer.name} is currently offline. An in-app alert has been sent to them.`);
      setTimeout(() => setOfflineCallAlert(''), 6000);
      return;
    }

    try {
      const res = await fetch(`${API}/api/staff-chat/calls`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({
          receiver_id: peer.id,
          call_type: callType,
        })
      });

      if (res.ok) {
        const callData = await res.json();
        if (callData.offline) {
          setOfflineCallAlert(callData.detail || `${peer.name} is currently offline. An in-app alert has been sent to them.`);
          setTimeout(() => setOfflineCallAlert(''), 6000);
          return;
        }
        window.dispatchEvent(new CustomEvent('recovera_start_staff_call', {
          detail: {
            callId: callData.call_id,
            isCaller: true,
            callType: callType,
            peerUser: peer,
            collisionResolved: callData.collision_resolved || false,
          }
        }));
      }
    } catch (err) {
      console.error('Initiate call error:', err);
    }
  };

  const handleAcceptIncomingCall = () => {
    if (!incomingCallPrompt) return;
    stopIncomingSound();
    const { callId, callType, caller } = incomingCallPrompt;
    wsRef.current?.send(JSON.stringify({
      type: 'call_accept',
      target_user_id: caller.id,
      call_id: callId,
    }));
    window.dispatchEvent(new CustomEvent('recovera_start_staff_call', {
      detail: {
        callId,
        isCaller: false,
        callType,
        peerUser: caller,
      }
    }));
    setIncomingCallPrompt(null);
  };

  const handleDeclineIncomingCall = () => {
    if (!incomingCallPrompt) return;
    stopIncomingSound();
    const { callId, caller } = incomingCallPrompt;
    wsRef.current?.send(JSON.stringify({
      type: 'call_reject',
      target_user_id: caller.id,
      call_id: callId,
    }));
    setIncomingCallPrompt(null);
  };

  // Start new chat with staff user
  const handleStartDirectChat = async (targetUserId) => {
    // Instant switch if conversation already exists in state
    const existing = conversations.find((c) => {
      const p = c.other_participant || c.other_user;
      return p?.id === targetUserId;
    });
    if (existing) {
      setActiveConvId(existing.id);
      setNewChatModalOpen(false);
      return;
    }

    try {
      const res = await fetch(`${API}/api/staff-chat/conversations/direct`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({ target_user_id: targetUserId })
      });

      if (res.ok) {
        const conv = await res.json();
        setNewChatModalOpen(false);
        setConversations((prev) => {
          const idx = prev.findIndex((c) => c.id === conv.id);
          if (idx >= 0) {
            const next = [...prev];
            next[idx] = conv;
            return next;
          }
          return [conv, ...prev];
        });
        setActiveConvId(conv.id);
      }
    } catch (err) {
      console.error('Start chat error:', err);
    }
  };

  const handleClearChat = async () => {
    if (!activeConvId) return;
    if (!window.confirm('Clear your chat history for this conversation?')) return;
    try {
      await fetch(`${API}/api/staff-chat/conversations/${activeConvId}/clear`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` }
      });
      setMessages([]);
      setContactDrawerOpen(false);
    } catch (e) {}
  };

  // Set of peer IDs with whom the user already has an active conversation
  const existingPeerSet = useMemo(() => {
    const set = new Set();
    conversations.forEach((c) => {
      const p = c.other_participant || c.other_user;
      if (p?.id) set.add(p.id);
    });
    return set;
  }, [conversations]);

  const q = (searchQuery || '').toLowerCase().trim();

  // Category 1: Recent Chats (matching existing conversations)
  const matchingRecentConversations = useMemo(() => {
    return conversations.filter((c) => {
      const peer = c.other_participant || c.other_user;
      const isGroup = c.is_group || c.conversation_type === 'group';
      const title = (isGroup ? c.title : (peer?.name || c.title || 'Staff Member'))?.toLowerCase() || '';
      const dept = (peer?.specialization || peer?.role || '').toLowerCase();
      const matchesSearch = !q || title.includes(q) || dept.includes(q);

      if (!matchesSearch) return false;

      if (filterTab === 'unread') return (c.unread_count || 0) > 0;
      if (filterTab === 'doctors') return peer?.role === 'doctor';
      if (filterTab === 'nurses') return peer?.role === 'nurse';
      if (filterTab === 'admins') return peer?.role === 'admin';
      return true;
    });
  }, [conversations, q, filterTab]);

  // Category 2: Hospital Staff (New Chat) matching search who do NOT have an active chat yet
  const matchingNewStaff = useMemo(() => {
    if (!q) return []; // Only categorize when user is searching
    return staffDirectory.filter((staff) => {
      if (staff.id === currentUserId) return false;
      if (existingPeerSet.has(staff.id)) return false; // Already in Recent Chats

      const name = (staff.name || '').toLowerCase();
      const spec = (staff.specialization || '').toLowerCase();
      const role = (staff.role || '').toLowerCase();
      const matchesSearch = name.includes(q) || spec.includes(q) || role.includes(q);
      if (!matchesSearch) return false;

      if (filterTab === 'unread') return false;
      if (filterTab === 'doctors') return staff.role === 'doctor';
      if (filterTab === 'nurses') return staff.role === 'nurse';
      if (filterTab === 'admins') return staff.role === 'admin';
      return true;
    });
  }, [staffDirectory, currentUserId, existingPeerSet, q, filterTab]);

  const activeConv = conversations.find((c) => c.id === activeConvId);
  const activePeer = activeConv?.other_participant || activeConv?.other_user;

  // Deduplicate messages by id before rendering
  const uniqueMessages = useMemo(() => {
    const seen = new Set();
    return messages.filter((m) => {
      if (!m?.id) return true;
      if (seen.has(m.id)) return false;
      seen.add(m.id);
      return true;
    });
  }, [messages]);

  // Group messages by Date (Today, Yesterday, or exact date)
  const renderedMessageGroups = useMemo(() => {
    const groups = [];
    let lastDateSep = null;

    uniqueMessages.forEach((msg) => {
      const dateSep = formatMessageDateSeparator(msg.created_at);
      if (dateSep !== lastDateSep) {
        groups.push({ type: 'date_separator', label: dateSep, id: `sep_${msg.id}` });
        lastDateSep = dateSep;
      }
      groups.push({ type: 'message', data: msg });
    });

    return groups;
  }, [uniqueMessages]);

  const renderConversationItem = (conv) => {
    const isSelected = conv.id === activeConvId;
    const isGroup = conv.is_group || conv.conversation_type === 'group';
    const peer = conv.other_participant || conv.other_user;
    const title = isGroup ? (conv.title || 'Staff Group') : (peer?.name || conv.title || 'Staff Member');
    const sub = isGroup
      ? `Group • ${conv.participants_count || 2} members`
      : (peer?.specialization || (peer?.role === 'admin' ? 'Administration' : peer?.role) || 'Hospital Staff');
    const unread = conv.unread_count || 0;
    const lastMsg = conv.last_message;

    return (
      <div
        key={conv.id}
        className={`staff-conv-item ${isSelected ? 'active' : ''}`}
        onClick={() => setActiveConvId(conv.id)}
      >
        <StaffAvatar
          name={title}
          avatarUrl={isGroup ? null : peer?.avatar_url}
          status={!isGroup ? peer?.availability_status || 'offline' : null}
          size={44}
          fontSize={15}
        />

        <div className="staff-conv-meta">
          <div className="staff-conv-top">
            <span className="staff-conv-title">{title}</span>
            {lastMsg?.created_at && (
              <span className="staff-conv-time">
                {formatMessageTime(lastMsg.created_at)}
              </span>
            )}
          </div>

          <div className="staff-conv-sub">{sub}</div>

          <div className="staff-conv-bottom">
            <span className="staff-conv-preview">
              {typingMap[conv.id] ? (
                <span style={{ color: 'var(--p)', fontWeight: 600 }}>typing...</span>
              ) : (
                lastMsg?.content || 'No messages yet'
              )}
            </span>
            {unread > 0 && (
              <span className="staff-conv-unread-pill">{unread}</span>
            )}
          </div>
        </div>
      </div>
    );
  };

  const renderNewStaffItem = (staff) => {
    return (
      <div
        key={staff.id}
        className="staff-new-chat-item"
        onClick={() => handleStartDirectChat(staff.id)}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, minWidth: 0 }}>
          <StaffAvatar
            name={staff.name}
            avatarUrl={staff.avatar_url}
            status={staff.availability_status || 'offline'}
            size={42}
            fontSize={14}
          />
          <div style={{ minWidth: 0 }}>
            <div style={{ fontSize: 13.5, fontWeight: 700, color: 'var(--text)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {staff.name}
            </div>
            <div style={{ fontSize: 11.5, color: 'var(--p)', fontWeight: 600 }}>
              {staff.specialization || (staff.role === 'admin' ? 'Administration' : staff.role)}
            </div>
          </div>
        </div>

        <button
          type="button"
          className="btn btn-primary btn-sm"
          style={{ padding: '4px 12px', fontSize: 11.5, borderRadius: 14, flexShrink: 0 }}
          onClick={(e) => {
            e.stopPropagation();
            handleStartDirectChat(staff.id);
          }}
        >
          Chat
        </button>
      </div>
    );
  };

  return (
    <div className="page staff-chat-page-container">
      {/* 3-Section Container */}
      <div className={`staff-chat-layout ${activeConvId ? 'has-active-chat' : 'no-active-chat'}`}>
        {/* ========================================================= */}
        {/* PANEL 1: CONVERSATION LIST & SEARCH                       */}
        {/* ========================================================= */}
        <div className={`staff-chat-sidebar-pane ${activeConvId ? 'mobile-hidden' : ''}`}>
          {/* Header */}
          <div className="staff-chat-sidebar-header">
            <div>
              <h2 style={{ fontSize: 17, fontWeight: 800, color: 'var(--text)' }}>Staff Chat</h2>
              <p style={{ fontSize: 11, color: 'var(--p)', fontWeight: 600 }}>
                Secure messaging for hospital staff only
              </p>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <button
                type="button"
                className="icon-btn active"
                onClick={() => setNewChatModalOpen(true)}
                title="New Chat"
              >
                <Plus size={18} />
              </button>
            </div>
          </div>

          {/* Search bar */}
          <div className="staff-chat-search-wrap">
            <div style={{ position: 'relative', width: '100%' }}>
              <Search size={15} style={{ position: 'absolute', left: 12, top: 10, color: 'var(--text-muted)' }} />
              <input
                type="text"
                className="staff-chat-search-input"
                placeholder="Search staff by name, role, department..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
          </div>

          {/* Filter Pills: All, Unread, Doctors, Nurses, Admins */}
          <div className="staff-chat-pills-row">
            {[
              { id: 'all', label: 'All' },
              { id: 'unread', label: 'Unread' },
              { id: 'doctors', label: 'Doctors' },
              { id: 'nurses', label: 'Nurses' },
              { id: 'admins', label: 'Admins' },
            ].map((tab) => (
              <button
                key={tab.id}
                type="button"
                className={`staff-chat-pill ${filterTab === tab.id ? 'active' : ''}`}
                onClick={() => setFilterTab(tab.id)}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Conversation list */}
          <div className="staff-chat-conv-list">
            {q ? (
              // Search Mode: Categorized under Recent Chats vs New Chat
              <>
                {matchingRecentConversations.length > 0 && (
                  <div style={{ marginBottom: 12 }}>
                    <div className="staff-search-section-header">
                      <span>Recent Chats</span>
                      <span className="staff-search-section-badge">{matchingRecentConversations.length}</span>
                    </div>
                    {matchingRecentConversations.map((conv) => renderConversationItem(conv))}
                  </div>
                )}

                {matchingNewStaff.length > 0 && (
                  <div style={{ marginBottom: 12 }}>
                    <div className="staff-search-section-header">
                      <span>New Chat</span>
                      <span className="staff-search-section-badge">{matchingNewStaff.length}</span>
                    </div>
                    {matchingNewStaff.map((staff) => renderNewStaffItem(staff))}
                  </div>
                )}

                {matchingRecentConversations.length === 0 && matchingNewStaff.length === 0 && (
                  <div className="empty" style={{ margin: '40px 16px', padding: '24px 12px' }}>
                    <p>No staff or chats found matching "{searchQuery}"</p>
                  </div>
                )}
              </>
            ) : (
              // Normal Mode
              matchingRecentConversations.length === 0 ? (
                <div className="empty" style={{ margin: '40px 16px', padding: '24px 12px' }}>
                  <p style={{ marginBottom: 12 }}>No conversations found.</p>
                  <button
                    type="button"
                    className="btn btn-primary btn-sm"
                    onClick={() => setNewChatModalOpen(true)}
                  >
                    <Plus size={14} /> Start New Chat
                  </button>
                </div>
              ) : (
                matchingRecentConversations.map((conv) => renderConversationItem(conv))
              )
            )}
          </div>
        </div>

        {/* ========================================================= */}
        {/* PANEL 2: WHATSAPP-STYLE CHAT VIEWPORT                     */}
        {/* ========================================================= */}
        <div className={`staff-chat-main-pane ${!activeConvId ? 'mobile-hidden' : ''}`}>
          {activeConv ? (
            <>
              {/* Top Navigation Bar */}
              <div className="staff-chat-navbar">
                {/* Mobile Back Button to return to chats list */}
                <button
                  type="button"
                  className="staff-chat-back-btn"
                  onClick={() => setActiveConvId(null)}
                  title="Back to conversations"
                >
                  <ChevronLeft size={22} />
                </button>

                <div
                  style={{ display: 'flex', alignItems: 'center', gap: 12, cursor: 'pointer', flex: 1, minWidth: 0 }}
                  onClick={() => setContactDrawerOpen(true)}
                  title="View contact info"
                >
                  <StaffAvatar
                    name={(activeConv.is_group || activeConv.conversation_type === 'group') ? '👥' : (activePeer?.name || activeConv.title || 'Staff Member')}
                    avatarUrl={(activeConv.is_group || activeConv.conversation_type === 'group') ? null : activePeer?.avatar_url}
                    status={!(activeConv.is_group || activeConv.conversation_type === 'group') ? (activePeer?.availability_status || 'offline') : null}
                    size={40}
                    fontSize={14}
                  />

                  <div style={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
                    <div style={{ fontSize: 14.5, fontWeight: 800, color: 'var(--text, #0f172a)' }}>
                      {(activeConv.is_group || activeConv.conversation_type === 'group')
                        ? (activeConv.title || 'Staff Group')
                        : (activePeer?.name || activeConv.title || 'Staff Member')}
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted, #64748b)', display: 'flex', alignItems: 'center', gap: 6 }}>
                      <span>
                        {(activeConv.is_group || activeConv.conversation_type === 'group')
                          ? `${activeConv.participants_count || 2} members`
                          : (activePeer?.specialization || (activePeer?.role === 'admin' ? 'Administration' : activePeer?.role) || 'Hospital Staff')}
                      </span>
                      {!(activeConv.is_group || activeConv.conversation_type === 'group') && (
                        <span style={{ color: activePeer?.availability_status === 'online' ? '#10b981' : '#94a3b8' }}>
                          • {activePeer?.availability_status ? activePeer.availability_status.toUpperCase() : 'OFFLINE'}
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                {/* Navbar Action Icons */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <button
                    type="button"
                    className="staff-composer-btn"
                    onClick={() => setSearchDrawerOpen(true)}
                    title="Search in conversation"
                  >
                    <Search size={18} />
                  </button>

                  {!activeConv.is_group && (
                    <>
                      <button
                        type="button"
                        className="staff-composer-btn"
                        onClick={() => handleInitiateCall('video')}
                        title="Video Call"
                      >
                        <Video size={18} />
                      </button>
                      <button
                        type="button"
                        className="staff-composer-btn"
                        onClick={() => handleInitiateCall('voice')}
                        title="Voice Call"
                      >
                        <Phone size={18} />
                      </button>
                    </>
                  )}

                  <button
                    type="button"
                    className="staff-composer-btn"
                    onClick={() => setContactDrawerOpen(!contactDrawerOpen)}
                    title="Contact Details"
                  >
                    <MoreVertical size={18} />
                  </button>
                </div>
              </div>

              {/* Chat Message Scrollport */}
              <div className="staff-chat-body" ref={chatBodyRef}>
                {renderedMessageGroups.map((item) => {
                  if (item.type === 'date_separator') {
                    return (
                      <div key={item.id} className="staff-date-separator">
                        {item.label}
                      </div>
                    );
                  }

                  const msg = item.data;
                  const isOutgoing = msg.sender_id === currentUserId;
                  const isDeleted = msg.is_deleted_for_everyone;

                  // Call event banner
                  if (msg.message_type === 'call_event') {
                    return (
                      <div key={msg.id} className="staff-call-event-bubble">
                        <Phone size={14} color="var(--p)" />
                        <span>{msg.content}</span>
                        <span style={{ fontSize: 10, opacity: 0.7 }}>{formatMessageTime(msg.created_at)}</span>
                      </div>
                    );
                  }

                  return (
                    <div
                      key={msg.id}
                      id={`msg-${msg.id}`}
                      className={`staff-bubble ${isOutgoing ? 'staff-bubble-out' : 'staff-bubble-in'} ${activeContextMenuMsgId === msg.id ? 'has-menu' : ''}`}
                    >
                      {/* Sender name for group chats */}
                      {activeConv.is_group && !isOutgoing && (
                        <div className="staff-bubble-sender">{msg.sender_name}</div>
                      )}

                      {/* Quoted reply snippet */}
                      {msg.reply_to && (
                        <div className="staff-bubble-reply-preview">
                          <div style={{ fontWeight: 700, color: 'var(--p)' }}>
                            {msg.reply_to.sender_name}
                          </div>
                          <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            {msg.reply_to.content}
                          </div>
                        </div>
                      )}

                      {/* Message Content: Voice or Text */}
                      {isDeleted ? (
                        <span style={{ fontStyle: 'italic', opacity: 0.75 }}>
                          🚫 This message was deleted
                        </span>
                      ) : msg.message_type === 'voice' && msg.voice_url ? (
                        <StaffVoicePlayer
                          voiceUrl={msg.voice_url.startsWith('http') ? msg.voice_url : `${API}${msg.voice_url}`}
                          duration={msg.voice_duration}
                          isOutgoing={isOutgoing}
                        />
                      ) : (
                        <span>{msg.content}</span>
                      )}

                      {/* Time and Read Ticks */}
                      <div className="staff-bubble-time-row">
                        <span className="staff-bubble-time">
                          {formatMessageTime(msg.created_at)}
                        </span>
                        {isOutgoing && (
                          <span className="staff-bubble-ticks">
                            <CheckCheck size={14} />
                          </span>
                        )}
                      </div>

                      {/* Context menu trigger icon */}
                      {!isDeleted && (
                        <div
                          className="staff-bubble-action-trigger"
                          onClick={(e) => {
                            e.stopPropagation();
                            setActiveContextMenuMsgId(activeContextMenuMsgId === msg.id ? null : msg.id);
                          }}
                        >
                          ▼
                        </div>
                      )}

                      {/* Context Menu Dropdown */}
                      {activeContextMenuMsgId === msg.id && (
                        <div
                          className={`staff-msg-context-menu ${isOutgoing ? 'outgoing' : 'incoming'}`}
                          onClick={(e) => e.stopPropagation()}
                        >
                          <div
                            className="staff-context-item"
                            onClick={() => {
                              setReplyingTo(msg);
                              setActiveContextMenuMsgId(null);
                            }}
                          >
                            <Reply size={14} /> Reply
                          </div>
                          <div
                            className="staff-context-item"
                            onClick={() => handleCopyText(msg.content)}
                          >
                            <Copy size={14} /> Copy
                          </div>
                          {isOutgoing && msg.message_type === 'text' && (
                            <div
                              className="staff-context-item"
                              onClick={() => {
                                setEditingMsg(msg);
                                setInputText(msg.content);
                                setActiveContextMenuMsgId(null);
                              }}
                            >
                              <Edit2 size={14} /> Edit
                            </div>
                          )}
                          <div
                            className="staff-context-item danger"
                            onClick={() => handleDeleteForMe(msg.id)}
                          >
                            <Trash2 size={14} /> Delete for me
                          </div>
                          {isOutgoing && (
                            <div
                              className="staff-context-item danger"
                              onClick={() => handleDeleteForEveryone(msg.id)}
                            >
                              <Trash2 size={14} /> Delete for everyone
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}

                {/* Real-time Typing Indicator */}
                {typingMap[activeConvId] && (
                  <div className="staff-bubble staff-bubble-in" style={{ padding: '6px 12px', fontSize: 12, color: '#94a3b8' }}>
                    <span style={{ fontWeight: 600, color: 'var(--p)' }}>{typingMap[activeConvId]}</span> is typing...
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>

              {/* Replying or Editing Banner */}
              {(replyingTo || editingMsg) && (
                <div style={{ padding: '8px 18px', background: 'rgba(0,0,0,0.4)', borderTop: '1px solid rgba(255,255,255,0.08)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div style={{ fontSize: 12, color: 'var(--p)' }}>
                    {editingMsg ? (
                      <span>Editing message: <strong>{editingMsg.content}</strong></span>
                    ) : (
                      <span>Replying to <strong>{replyingTo.sender_name}</strong>: {replyingTo.content}</span>
                    )}
                  </div>
                  <button
                    type="button"
                    className="staff-composer-btn"
                    style={{ width: 24, height: 24 }}
                    onClick={() => {
                      setReplyingTo(null);
                      setEditingMsg(null);
                      setInputText('');
                    }}
                  >
                    <X size={14} />
                  </button>
                </div>
              )}

              {/* Message Composer (Emoji, Text, Voice, Send - NO File Attachments) */}
              <div className="staff-composer-container">
                {isVoiceRecording ? (
                  <StaffVoiceRecorder
                    onSendVoice={handleSendVoice}
                    onCancel={() => setIsVoiceRecording(false)}
                  />
                ) : (
                  <form onSubmit={handleSendMessage} className="staff-composer-row">
                    {/* Emoji Picker toggle */}
                    <div style={{ position: 'relative' }}>
                      <button
                        type="button"
                        className="staff-composer-btn"
                        onClick={() => setEmojiPickerOpen(!emojiPickerOpen)}
                        title="Emoji"
                      >
                        <Smile size={20} />
                      </button>

                      {emojiPickerOpen && (
                        <div
                          style={{
                            position: 'absolute',
                            bottom: 50,
                            left: 0,
                            background: '#233138',
                            border: '1px solid rgba(255,255,255,0.1)',
                            borderRadius: 10,
                            padding: 8,
                            display: 'grid',
                            gridTemplateColumns: 'repeat(6, 1fr)',
                            gap: 6,
                            zIndex: 40,
                            boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
                          }}
                        >
                          {QUICK_EMOJIS.map((emoji) => (
                            <button
                              key={emoji}
                              type="button"
                              onClick={() => {
                                setInputText((prev) => prev + emoji);
                                setEmojiPickerOpen(false);
                              }}
                              style={{
                                fontSize: 20,
                                background: 'transparent',
                                border: 'none',
                                cursor: 'pointer',
                                padding: 4,
                              }}
                            >
                              {emoji}
                            </button>
                          ))}
                        </div>
                      )}
                    </div>

                    {/* Text input */}
                    <input
                      type="text"
                      className="staff-composer-input"
                      placeholder="Type a message..."
                      value={inputText}
                      onChange={handleInputChange}
                    />

                    {/* Mic button (voice record) or Send button */}
                    {inputText.trim() ? (
                      <button
                        type="submit"
                        className="staff-composer-btn staff-composer-send-btn"
                        title="Send message"
                      >
                        <Send size={18} />
                      </button>
                    ) : (
                      <button
                        type="button"
                        className="staff-composer-btn staff-composer-send-btn"
                        onClick={() => setIsVoiceRecording(true)}
                        title="Record voice message"
                      >
                        <Mic size={18} />
                      </button>
                    )}
                  </form>
                )}
              </div>
            </>
          ) : (
            <div className="empty" style={{ margin: 'auto' }}>
              <div className="empty-icon">
                <Shield size={32} />
              </div>
              <div className="empty-title">Recovera AI Staff Messaging</div>
              <p className="empty-desc">
                Select a staff member from the left panel or click '+' to start a new chat.
              </p>
            </div>
          )}
        </div>

        {/* ========================================================= */}
        {/* PANEL 3: SLIDE-OVER SEARCH IN CONVERSATION DRAWER         */}
        {/* ========================================================= */}
        {searchDrawerOpen && activeConvId && (
          <StaffSearchDrawer
            conversationId={activeConvId}
            token={token}
            onClose={() => setSearchDrawerOpen(false)}
            onSelectResult={(msgId) => {
              const el = document.getElementById(`msg-${msgId}`);
              if (el) {
                el.scrollIntoView({ behavior: 'smooth', block: 'center' });
                el.style.outline = '2px solid var(--p)';
                setTimeout(() => {
                  el.style.outline = 'none';
                }, 2000);
              }
            }}
          />
        )}

        {/* ========================================================= */}
        {/* PANEL 4: SLIDE-OVER CONTACT INFO DRAWER                   */}
        {/* ========================================================= */}
        {contactDrawerOpen && activePeer && (
          <StaffContactInfoDrawer
            contact={activePeer}
            onClose={() => setContactDrawerOpen(false)}
            onStartVoiceCall={() => handleInitiateCall('voice')}
            onStartVideoCall={() => handleInitiateCall('video')}
            onOpenSearch={() => {
              setContactDrawerOpen(false);
              setSearchDrawerOpen(true);
            }}
            onClearChat={handleClearChat}
            onToggleMute={() => {}}
            isMuted={false}
          />
        )}
      </div>

      {/* New Chat Modal */}
      {newChatModalOpen && (
        <StaffNewChatModal
          staffDirectory={staffDirectory}
          existingConversations={conversations}
          currentUserId={currentUserId}
          onClose={() => setNewChatModalOpen(false)}
          onStartChat={handleStartDirectChat}
        />
      )}

      {/* Global Staff Call Modal is hosted at AppShell level for seamless PIP across routes */}

      {/* Offline Call In-App Alert Toast */}
      {offlineCallAlert && (
        <div style={{
          position: 'fixed',
          top: 24,
          left: '50%',
          transform: 'translateX(-50%)',
          zIndex: 99999,
          background: '#0f172a',
          color: '#f8fafc',
          padding: '12px 24px',
          borderRadius: 10,
          border: '1.5px solid #38bdf8',
          boxShadow: '0 12px 36px rgba(0,0,0,0.6)',
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          fontSize: 13.5,
          fontWeight: 600,
          animation: 'fade-in 0.2s ease-out'
        }}>
          <AlertCircle size={18} color="#38bdf8" style={{ flexShrink: 0 }} />
          <span>{offlineCallAlert}</span>
          <button
            type="button"
            onClick={() => setOfflineCallAlert('')}
            style={{
              background: 'none',
              border: 'none',
              color: '#94a3b8',
              cursor: 'pointer',
              display: 'grid',
              placeItems: 'center',
              padding: 4
            }}
          >
            <X size={15} />
          </button>
        </div>
      )}

      {/* Incoming Call Dialog with Audio Ringtone */}
      {incomingCallPrompt && (
        <div style={{
          position: 'fixed',
          inset: 0,
          background: 'rgba(0, 0, 0, 0.72)',
          backdropFilter: 'blur(4px)',
          zIndex: 99999,
          display: 'grid',
          placeItems: 'center',
          padding: 20
        }}>
          <div style={{
            background: '#111b21',
            borderRadius: 16,
            border: '1px solid rgba(255, 255, 255, 0.12)',
            boxShadow: '0 20px 50px rgba(0, 0, 0, 0.8)',
            padding: '36px 28px',
            width: '100%',
            maxWidth: 340,
            textAlign: 'center',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: 16
          }}>
            <div style={{ position: 'relative' }}>
              <StaffAvatar
                name={incomingCallPrompt.caller?.name}
                avatarUrl={incomingCallPrompt.caller?.avatar_url}
                size={88}
                fontSize={30}
              />
              <span
                className="status-dot online pulse"
                style={{
                  position: 'absolute',
                  bottom: 2,
                  right: 2,
                  width: 14,
                  height: 14
                }}
              />
            </div>

            <div>
              <h3 style={{ fontSize: 19, fontWeight: 800, color: '#f8fafc', marginBottom: 4 }}>
                {incomingCallPrompt.caller?.name}
              </h3>
              <p style={{ fontSize: 12.5, color: 'var(--p)', fontWeight: 600, textTransform: 'capitalize' }}>
                {incomingCallPrompt.caller?.specialization || incomingCallPrompt.caller?.role}
              </p>
              <p style={{ fontSize: 12, color: '#94a3b8', marginTop: 6 }}>
                Incoming {incomingCallPrompt.callType === 'video' ? 'Video' : 'Voice'} Call...
              </p>
            </div>

            <div style={{ display: 'flex', gap: 32, marginTop: 8 }}>
              {/* Decline Button */}
              <button
                type="button"
                onClick={handleDeclineIncomingCall}
                style={{
                  width: 54,
                  height: 54,
                  borderRadius: '50%',
                  background: '#ef4444',
                  color: '#ffffff',
                  border: 'none',
                  display: 'grid',
                  placeItems: 'center',
                  cursor: 'pointer',
                  boxShadow: '0 4px 16px rgba(239, 68, 68, 0.45)',
                  transition: 'transform 0.15s ease'
                }}
                title="Decline Call"
              >
                <PhoneOff size={24} />
              </button>

              {/* Accept Button */}
              <button
                type="button"
                onClick={handleAcceptIncomingCall}
                style={{
                  width: 54,
                  height: 54,
                  borderRadius: '50%',
                  background: '#10b981',
                  color: '#ffffff',
                  border: 'none',
                  display: 'grid',
                  placeItems: 'center',
                  cursor: 'pointer',
                  boxShadow: '0 4px 16px rgba(16, 185, 129, 0.45)',
                  transition: 'transform 0.15s ease'
                }}
                title="Accept Call"
              >
                <Phone size={24} />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
