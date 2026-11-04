import React from 'react';
import { X, Phone, Video, Search, Bell, BellOff, Trash2, Ban, Shield, Mail, PhoneCall } from 'lucide-react';
import { StaffAvatar } from './StaffAvatar';

export function StaffContactInfoDrawer({
  contact,
  onClose,
  onStartVoiceCall,
  onStartVideoCall,
  onOpenSearch,
  onClearChat,
  onToggleMute,
  isMuted = false,
}) {
  if (!contact) return null;

  return (
    <div className="staff-contact-drawer">
      {/* Header */}
      <div style={{ padding: '14px 18px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid var(--border)' }}>
        <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--text)' }}>Contact Info</span>
        <button type="button" className="staff-composer-btn" onClick={onClose}>
          <X size={18} />
        </button>
      </div>

      <div style={{ padding: '24px 20px', display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center', borderBottom: '1px solid var(--border)' }}>
        <StaffAvatar
          name={contact.name}
          avatarUrl={contact.avatar_url}
          status={contact.availability_status || 'offline'}
          size={84}
          fontSize={28}
          style={{ marginBottom: 12 }}
        />

        <h3 style={{ fontSize: 18, fontWeight: 800, color: 'var(--text)', marginBottom: 2 }}>
          {contact.name}
        </h3>
        <p style={{ fontSize: 12.5, color: 'var(--p)', fontWeight: 600, marginBottom: 6 }}>
          {contact.specialization || contact.role}
        </p>
        <span className={`badge badge-${contact.availability_status === 'online' ? 'green' : 'amber'}`}>
          • {contact.availability_status ? contact.availability_status.toUpperCase() : 'OFFLINE'}
        </span>

        {/* Quick Call & Search Buttons */}
        <div style={{ display: 'flex', gap: 12, marginTop: 18, width: '100%', justifyContent: 'center' }}>
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={onStartVoiceCall}
            style={{ flex: 1 }}
          >
            <Phone size={14} /> Voice Call
          </button>
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={onStartVideoCall}
            style={{ flex: 1 }}
          >
            <Video size={14} /> Video Call
          </button>
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={onOpenSearch}
            title="Search in chat"
          >
            <Search size={14} />
          </button>
        </div>
      </div>

      {/* About & Contact Details */}
      <div style={{ padding: '18px 20px', display: 'flex', flexDirection: 'column', gap: 14, borderBottom: '1px solid var(--border)' }}>
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>
            About
          </div>
          <div style={{ fontSize: 13, color: 'var(--text)', lineHeight: 1.45 }}>
            {contact.about || 'Available for patient consultations and hospital coordination.'}
          </div>
        </div>

        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>
            Department
          </div>
          <div style={{ fontSize: 13, color: 'var(--text)', display: 'flex', alignItems: 'center', gap: 6 }}>
            <span>{contact.specialization || 'Clinical Operations'}</span>
            <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>(Read-only)</span>
          </div>
        </div>

        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>
            Role
          </div>
          <div style={{ fontSize: 13, color: 'var(--text)', textTransform: 'capitalize' }}>
            {contact.role}
          </div>
        </div>

        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>
            Email
          </div>
          <div style={{ fontSize: 12.5, color: 'var(--text-dim)', display: 'flex', alignItems: 'center', gap: 6 }}>
            <Mail size={13} /> {contact.email}
          </div>
        </div>

        {contact.phone && (
          <div>
            <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>
              Phone
            </div>
            <div style={{ fontSize: 12.5, color: 'var(--text-dim)', display: 'flex', alignItems: 'center', gap: 6 }}>
              <PhoneCall size={13} /> {contact.phone}
            </div>
          </div>
        )}
      </div>

      {/* Options: Mute, Clear Chat, Block */}
      <div style={{ padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: 6 }}>
        <button
          type="button"
          onClick={onToggleMute}
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '10px 12px',
            borderRadius: 'var(--radius-sm)',
            background: 'transparent',
            border: 'none',
            color: 'var(--text)',
            cursor: 'pointer',
            fontSize: 13,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            {isMuted ? <BellOff size={16} color="var(--warn)" /> : <Bell size={16} />}
            <span>Mute notifications</span>
          </div>
          <span style={{ fontSize: 11, fontWeight: 700, color: isMuted ? 'var(--warn)' : 'var(--text-muted)' }}>
            {isMuted ? 'ON' : 'OFF'}
          </span>
        </button>

        <button
          type="button"
          onClick={onClearChat}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            padding: '10px 12px',
            borderRadius: 'var(--radius-sm)',
            background: 'transparent',
            border: 'none',
            color: 'var(--text-dim)',
            cursor: 'pointer',
            fontSize: 13,
            textAlign: 'left',
          }}
        >
          <Trash2 size={16} />
          <span>Clear chat history</span>
        </button>

        <button
          type="button"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            padding: '10px 12px',
            borderRadius: 'var(--radius-sm)',
            background: 'transparent',
            border: 'none',
            color: '#ef4444',
            cursor: 'pointer',
            fontSize: 13,
            textAlign: 'left',
          }}
        >
          <Ban size={16} />
          <span>Block user</span>
        </button>
      </div>
    </div>
  );
}
