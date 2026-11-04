import React, { useState } from 'react';
import { X, Camera, Lock, CheckCircle2, User, Bell } from 'lucide-react';
import { StaffAvatar } from './StaffAvatar';

const AVATAR_PRESETS = [
  'https://images.unsplash.com/photo-1594824813637-b67272714c6e?w=400&auto=format&fit=crop&q=80',
  'https://images.unsplash.com/photo-1622253692010-333f2da6031d?w=400&auto=format&fit=crop&q=80',
  'https://images.unsplash.com/photo-1582750433449-648ed127bb54?w=400&auto=format&fit=crop&q=80',
  'https://images.unsplash.com/photo-1559839734-2b71ea197ec2?w=400&auto=format&fit=crop&q=80',
  'https://images.unsplash.com/photo-1537368910025-700350fe46c7?w=400&auto=format&fit=crop&q=80',
  'https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=400&auto=format&fit=crop&q=80',
  'https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=400&auto=format&fit=crop&q=80',
];

export function StaffProfileModal({ profile, token, onClose, onUpdated }) {
  const [avatarUrl, setAvatarUrl] = useState(profile.avatar_url || '');
  const [about, setAbout] = useState(profile.about || '');
  const [availability, setAvailability] = useState(profile.availability_status || 'online');
  const [muteNotifications, setMuteNotifications] = useState(profile.mute_notifications || false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSaving(true);
    setError(null);

    try {
      const res = await fetch('/api/staff-chat/profile', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          avatar_url: avatarUrl,
          about: about,
          availability_status: availability,
          mute_notifications: muteNotifications,
        }),
      });

      if (!res.ok) {
        throw new Error('Failed to update profile');
      }

      const updated = await res.json();
      onUpdated(updated);
      onClose();
    } catch (err) {
      setError(err.message || 'Error updating profile');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="modal-backdrop">
      <div className="modal modal-md">
        <div className="modal-head">
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div className="icon-btn" style={{ background: 'var(--p-dim)', color: 'var(--p)' }}>
              <User size={18} />
            </div>
            <div>
              <h3 className="modal-title" style={{ fontSize: 17 }}>Staff Profile (Edit)</h3>
              <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                Customize your avatar, availability status, and bio quote
              </p>
            </div>
          </div>
          <button type="button" className="icon-btn" onClick={onClose}>
            <X size={18} />
          </button>
        </div>

        {error && (
          <div className="alert alert-error" style={{ marginBottom: 16 }}>
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          {/* Avatar Selection */}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginBottom: 20 }}>
            <StaffAvatar
              name={profile.name}
              avatarUrl={avatarUrl}
              status={availability}
              size={80}
              fontSize={28}
              style={{ marginBottom: 12 }}
            />

            <div style={{ fontSize: 11.5, fontWeight: 700, color: 'var(--text-muted)', marginBottom: 8 }}>
              Choose a Hospital Profile Avatar:
            </div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'center' }}>
              {AVATAR_PRESETS.map((preset, idx) => (
                <img
                  key={idx}
                  src={preset}
                  alt={`preset-${idx}`}
                  onClick={() => setAvatarUrl(preset)}
                  style={{
                    width: 36,
                    height: 36,
                    borderRadius: '50%',
                    objectFit: 'cover',
                    cursor: 'pointer',
                    border: avatarUrl === preset ? '2px solid var(--p)' : '1px solid var(--border)',
                    transform: avatarUrl === preset ? 'scale(1.1)' : 'none',
                    transition: 'all 0.15s ease',
                  }}
                />
              ))}
            </div>
          </div>

          {/* Full Name (Read-only / info) */}
          <div className="form-field">
            <label className="form-label">Full Name</label>
            <input
              type="text"
              className="form-input"
              value={profile.name || ''}
              disabled
              style={{ opacity: 0.75, cursor: 'not-allowed' }}
            />
          </div>

          {/* About / Bio Quote */}
          <div className="form-field">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <label className="form-label">About / Status Message</label>
              <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{about.length}/160</span>
            </div>
            <textarea
              className="form-input"
              rows={2}
              maxLength={160}
              placeholder="e.g. Focused on patient care. Available for ward rounds."
              value={about}
              onChange={(e) => setAbout(e.target.value)}
            />
          </div>

          {/* Department / Specialty (Read-only for clinical integrity) */}
          <div className="form-field">
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
              <label className="form-label" style={{ marginBottom: 0 }}>Department / Specialty</label>
              <span style={{ fontSize: 11, color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 2 }}>
                <Lock size={11} /> (Read-only)
              </span>
            </div>
            <input
              type="text"
              className="form-input"
              value={profile.specialization || (profile.role === 'admin' ? 'Administration' : 'Clinical Operations')}
              disabled
              style={{ opacity: 0.75, cursor: 'not-allowed' }}
            />
          </div>

          {/* Role (Read-only) */}
          <div className="form-field">
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
              <label className="form-label" style={{ marginBottom: 0 }}>Role</label>
              <span style={{ fontSize: 11, color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 2 }}>
                <Lock size={11} /> (Read-only)
              </span>
            </div>
            <input
              type="text"
              className="form-input"
              value={profile.role?.toUpperCase() || ''}
              disabled
              style={{ opacity: 0.75, cursor: 'not-allowed' }}
            />
          </div>

          {/* Online Availability Status */}
          <div className="form-field">
            <label className="form-label">Online Availability Status</label>
            <select
              className="form-input"
              value={availability}
              onChange={(e) => setAvailability(e.target.value)}
            >
              <option value="online">🟢 Online</option>
              <option value="away">🟡 Away</option>
              <option value="busy">🔴 Busy / Do Not Disturb</option>
              <option value="offline">⚪ Offline</option>
            </select>
          </div>

          {/* Notification Mute Toggle */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 0', borderTop: '1px solid var(--border)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Bell size={16} color="var(--text-muted)" />
              <div>
                <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)' }}>Mute Notifications</div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Mute sound alerts for new staff chat messages</div>
              </div>
            </div>
            <input
              type="checkbox"
              checked={muteNotifications}
              onChange={(e) => setMuteNotifications(e.target.checked)}
              style={{ width: 18, height: 18, cursor: 'pointer', accentColor: 'var(--p)' }}
            />
          </div>

          {/* Modal Actions */}
          <div className="modal-footer">
            <button type="button" className="btn btn-ghost" onClick={onClose} disabled={saving}>
              Cancel
            </button>
            <button type="submit" className="btn btn-primary" disabled={saving}>
              {saving ? 'Saving...' : 'Save Changes'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
