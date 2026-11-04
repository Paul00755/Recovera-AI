import React, { useState, useMemo } from 'react';
import { Search, X, MessageSquare, Shield, Stethoscope, HeartPulse, Check } from 'lucide-react';
import { StaffAvatar } from './StaffAvatar';

export function StaffNewChatModal({
  staffDirectory = [],
  existingConversations = [],
  currentUserId,
  onClose,
  onStartChat
}) {
  const [searchTerm, setSearchTerm] = useState('');
  const [roleFilter, setRoleFilter] = useState('all');
  const [startingUserId, setStartingUserId] = useState(null);

  const existingPeerSet = useMemo(() => {
    const set = new Set();
    existingConversations.forEach((c) => {
      const p = c.other_participant || c.other_user;
      if (p?.id) set.add(p.id);
    });
    return set;
  }, [existingConversations]);

  const filtered = useMemo(() => {
    return staffDirectory.filter((staff) => {
      // Never show current user to themselves
      if (currentUserId && staff.id === currentUserId) return false;

      const matchesSearch =
        staff.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        (staff.specialization && staff.specialization.toLowerCase().includes(searchTerm.toLowerCase())) ||
        staff.role.toLowerCase().includes(searchTerm.toLowerCase());

      const matchesRole =
        roleFilter === 'all' ||
        (roleFilter === 'doctors' && staff.role === 'doctor') ||
        (roleFilter === 'nurses' && staff.role === 'nurse') ||
        (roleFilter === 'admins' && staff.role === 'admin');

      return matchesSearch && matchesRole;
    });
  }, [staffDirectory, currentUserId, searchTerm, roleFilter]);

  const recentChats = useMemo(() => filtered.filter((s) => existingPeerSet.has(s.id)), [filtered, existingPeerSet]);
  const newChats = useMemo(() => filtered.filter((s) => !existingPeerSet.has(s.id)), [filtered, existingPeerSet]);

  const handleClickChat = async (staffId) => {
    if (startingUserId) return; // Prevent double-triggering
    setStartingUserId(staffId);
    try {
      await onStartChat(staffId);
    } finally {
      setStartingUserId(null);
    }
  };

  const renderStaffRow = (staff, isRecent) => {
    const isStarting = startingUserId === staff.id;
    return (
      <div
        key={staff.id}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '10px 14px',
          borderRadius: 'var(--radius-md)',
          background: 'var(--surface2)',
          border: '1px solid var(--border)',
          transition: 'background 0.15s ease',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <StaffAvatar
            name={staff.name}
            avatarUrl={staff.avatar_url}
            status={staff.availability_status || 'offline'}
            size={42}
            fontSize={14}
          />

          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 13.5, fontWeight: 700, color: 'var(--text)' }}>
                {staff.name}
              </span>
              {isRecent && (
                <span
                  style={{
                    fontSize: 10.5,
                    fontWeight: 700,
                    padding: '1px 7px',
                    borderRadius: 10,
                    background: 'var(--p-dim)',
                    color: 'var(--p)',
                    border: '1px solid var(--p-border)',
                  }}
                >
                  Existing
                </span>
              )}
            </div>
            <div style={{ fontSize: 11.5, color: 'var(--p)', fontWeight: 600 }}>
              {staff.specialization || (staff.role === 'admin' ? 'Administration' : staff.role)}
            </div>
          </div>
        </div>

        <button
          type="button"
          className={`btn btn-sm ${isRecent ? 'btn-secondary' : 'btn-primary'}`}
          disabled={isStarting}
          onClick={() => handleClickChat(staff.id)}
        >
          {isStarting ? 'Opening...' : isRecent ? 'Open Chat' : 'Chat'}
        </button>
      </div>
    );
  };

  return (
    <div className="modal-backdrop">
      <div className="modal modal-md">
        <div className="modal-head">
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div className="icon-btn" style={{ background: 'var(--p-dim)', color: 'var(--p)' }}>
              <MessageSquare size={18} />
            </div>
            <div>
              <h3 className="modal-title" style={{ fontSize: 17 }}>New Chat</h3>
              <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                Start a private conversation with any hospital staff member
              </p>
            </div>
          </div>
          <button type="button" className="icon-btn" onClick={onClose}>
            <X size={18} />
          </button>
        </div>

        {/* Search Bar */}
        <div style={{ position: 'relative', marginBottom: 12 }}>
          <Search size={16} style={{ position: 'absolute', left: 14, top: 12, color: 'var(--text-muted)' }} />
          <input
            type="text"
            className="form-input"
            style={{ paddingLeft: 38 }}
            placeholder="Search by name, role, department..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            autoFocus
          />
        </div>

        {/* Role Filter Tabs */}
        <div className="staff-chat-pills-row" style={{ padding: '0 0 14px 0' }}>
          {[
            { id: 'all', label: 'All' },
            { id: 'doctors', label: 'Doctors' },
            { id: 'nurses', label: 'Nurses' },
            { id: 'admins', label: 'Admins' },
          ].map((tab) => (
            <button
              key={tab.id}
              type="button"
              className={`staff-chat-pill ${roleFilter === tab.id ? 'active' : ''}`}
              onClick={() => setRoleFilter(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Categorized Staff List */}
        <div style={{ maxHeight: 380, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 10 }}>
          {filtered.length === 0 ? (
            <div className="empty" style={{ padding: '32px 16px' }}>
              <p>No hospital staff found matching your search.</p>
            </div>
          ) : (
            <>
              {recentChats.length > 0 && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  <div className="staff-search-section-header">
                    <span>Recent Chats</span>
                    <span className="staff-search-section-badge">{recentChats.length}</span>
                  </div>
                  {recentChats.map((staff) => renderStaffRow(staff, true))}
                </div>
              )}

              {newChats.length > 0 && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  <div className="staff-search-section-header">
                    <span>New Chat</span>
                    <span className="staff-search-section-badge">{newChats.length}</span>
                  </div>
                  {newChats.map((staff) => renderStaffRow(staff, false))}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
