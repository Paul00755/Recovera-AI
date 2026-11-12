import React, { useState, useEffect, useRef } from 'react';
import { Plus, Bell, Shield, AlertTriangle, BookOpen, Calendar, Users, Edit3, Trash2, CheckCircle, Search, Filter, Image as ImageIcon, X, Upload } from 'lucide-react';
import { StaffAvatar } from './StaffAvatar';
import { API } from './staffChatUtils';

const CATEGORIES = [
  { id: 'all', label: 'All' },
  { id: 'general', label: 'General' },
  { id: 'emergency', label: 'Emergency' },
  { id: 'policy', label: 'Policy' },
  { id: 'training', label: 'Training' },
  { id: 'scheduling', label: 'Scheduling' },
  { id: 'new_staff', label: 'New Staff' },
];

export function StaffAnnouncementsPage({ session, navigate }) {
  const [announcements, setAnnouncements] = useState([]);
  const [loading, setLoading] = useState(true);
  const [categoryFilter, setCategoryFilter] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [modalOpen, setModalOpen] = useState(false);
  const [editingAnn, setEditingAnn] = useState(null);
  const [unreadCount, setUnreadCount] = useState(0);

  // Form state
  const [formTitle, setFormTitle] = useState('');
  const [formContent, setFormContent] = useState('');
  const [formCategory, setFormCategory] = useState('general');
  const [formPriority, setFormPriority] = useState('normal');
  const [formImageUrl, setFormImageUrl] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const fileInputRef = useRef(null);

  const isAdmin = session?.user?.role === 'admin';
  const token = session?.token || JSON.parse(sessionStorage.getItem('recovera_session') || localStorage.getItem('recovera_session') || '{}')?.token;

  useEffect(() => {
    loadAnnouncements();
  }, [categoryFilter, searchTerm]);

  const loadAnnouncements = async () => {
    setLoading(true);
    try {
      let url = `${API}/api/announcements?category=${categoryFilter}`;
      if (searchTerm.trim()) {
        url += `&search=${encodeURIComponent(searchTerm.trim())}`;
      }
      const res = await fetch(url, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setAnnouncements(data);
        const unread = data.filter((a) => !a.is_read).length;
        setUnreadCount(unread);
      }
    } catch (e) {
      console.error('Failed to load announcements:', e);
    } finally {
      setLoading(false);
    }
  };

  const handleMarkAsRead = async (annId) => {
    try {
      await fetch(`${API}/api/announcements/${annId}/read`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` }
      });
      setAnnouncements((prev) =>
        prev.map((a) => (a.id === annId ? { ...a, is_read: true } : a))
      );
      setUnreadCount((c) => Math.max(0, c - 1));
    } catch (e) {}
  };

  const handleMarkAllRead = async () => {
    try {
      await fetch(`${API}/api/announcements/mark-all-read`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` }
      });
      setAnnouncements((prev) => prev.map((a) => ({ ...a, is_read: true })));
      setUnreadCount(0);
    } catch (e) {}
  };

  const handleOpenCreate = () => {
    setEditingAnn(null);
    setFormTitle('');
    setFormContent('');
    setFormCategory('general');
    setFormPriority('normal');
    setFormImageUrl('');
    setSelectedFile(null);
    setImagePreview(null);
    setError('');
    setModalOpen(true);
  };

  const handleOpenEdit = (ann) => {
    setEditingAnn(ann);
    setFormTitle(ann.title);
    setFormContent(ann.content);
    setFormCategory(ann.category);
    setFormPriority(ann.priority);
    setFormImageUrl(ann.image_url || '');
    setSelectedFile(null);
    const existingImg = ann.image_url
      ? (ann.image_url.startsWith('http') || ann.image_url.startsWith('blob:') ? ann.image_url : `${API}${ann.image_url}`)
      : null;
    setImagePreview(existingImg);
    setError('');
    setModalOpen(true);
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      setError('Please select a valid image file (PNG, JPG, WEBP, GIF).');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setError('Selected image exceeds the 10MB limit.');
      return;
    }
    setError('');
    setSelectedFile(file);
    setImagePreview(URL.createObjectURL(file));
  };

  const handleRemoveImage = () => {
    setSelectedFile(null);
    setImagePreview(null);
    setFormImageUrl('');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleDelete = async (annId) => {
    if (!window.confirm('Are you sure you want to delete this official announcement?')) return;
    try {
      const res = await fetch(`${API}/api/announcements/${annId}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` }
      });
      if (res.ok) {
        setAnnouncements((prev) => prev.filter((a) => a.id !== annId));
      }
    } catch (e) {
      console.error('Delete error:', e);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!formTitle.trim() || !formContent.trim()) {
      setError('Title and content are required.');
      return;
    }
    setSubmitting(true);
    setError('');

    try {
      let finalImageUrl = formImageUrl ? formImageUrl.trim() : null;

      // Upload selected image file to server if provided
      if (selectedFile) {
        const formData = new FormData();
        formData.append('file', selectedFile);
        const upRes = await fetch(`${API}/api/announcements/upload-image`, {
          method: 'POST',
          headers: { Authorization: `Bearer ${token}` },
          body: formData
        });
        if (!upRes.ok) {
          const errData = await upRes.json().catch(() => ({}));
          throw new Error(errData.detail || 'Failed to upload cover image.');
        }
        const upJson = await upRes.json();
        finalImageUrl = upJson.url;
      }

      const payload = {
        title: formTitle.trim(),
        content: formContent.trim(),
        category: formCategory,
        priority: formPriority,
        image_url: finalImageUrl,
      };

      let res;
      if (editingAnn) {
        res = await fetch(`${API}/api/announcements/${editingAnn.id}`, {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`
          },
          body: JSON.stringify(payload)
        });
      } else {
        res = await fetch(`${API}/api/announcements`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`
          },
          body: JSON.stringify(payload)
        });
      }

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Failed to save announcement');
      }

      setModalOpen(false);
      loadAnnouncements();
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="page">
      {/* Page Header */}
      <div className="section-head" style={{ marginBottom: 12 }}>
        <div>
          <span className="eyebrow">Hospital Staff Workspace</span>
          <h1 style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            Announcements
            {unreadCount > 0 && (
              <span className="badge badge-teal" style={{ fontSize: 11 }}>
                {unreadCount} Unread
              </span>
            )}
          </h1>
          <p style={{ marginTop: 4 }}>
            Official bulletins, clinical memos, and policy notices from hospital administration (Staff only).
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          {unreadCount > 0 && (
            <button type="button" className="btn btn-ghost btn-sm" onClick={handleMarkAllRead}>
              <CheckCircle size={14} /> Mark all read
            </button>
          )}

          {isAdmin && (
            <button type="button" className="btn btn-primary" onClick={handleOpenCreate}>
              <Plus size={16} /> New Announcement
            </button>
          )}
        </div>
      </div>

      {/* Filter Tabs and Search Bar */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 14, flexWrap: 'wrap' }}>
        <div className="staff-chat-pills-row" style={{ padding: 0 }}>
          {CATEGORIES.map((cat) => (
            <button
              key={cat.id}
              type="button"
              className={`staff-chat-pill ${categoryFilter === cat.id ? 'active' : ''}`}
              onClick={() => setCategoryFilter(cat.id)}
            >
              {cat.label}
            </button>
          ))}
        </div>

        <div style={{ position: 'relative', width: 260 }}>
          <Search size={15} style={{ position: 'absolute', left: 12, top: 11, color: 'var(--text-muted)' }} />
          <input
            type="text"
            className="form-input"
            style={{ paddingLeft: 34, paddingRight: 12, height: 36, fontSize: 13 }}
            placeholder="Search bulletins..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
      </div>

      {/* Announcements Grid */}
      {loading ? (
        <div className="empty" style={{ padding: '60px 24px' }}>
          <span className="spinner" />
          <p style={{ marginTop: 12 }}>Loading official bulletins...</p>
        </div>
      ) : announcements.length === 0 ? (
        <div className="empty" style={{ padding: '60px 24px' }}>
          <div className="empty-icon">
            <Bell size={28} />
          </div>
          <div className="empty-title">No announcements found</div>
          <p className="empty-desc">
            There are currently no announcements matching your selected category.
          </p>
        </div>
      ) : (
        <div className="announcements-grid">
          {announcements.map((ann) => {
            const isEmergency = ann.priority === 'emergency';
            const isImportant = ann.priority === 'important';

            return (
              <div
                key={ann.id}
                className={`announcement-card ${isEmergency ? 'emergency' : ''} ${!ann.is_read ? 'unread' : ''}`}
              >
                {/* Hero image if exists */}
                {ann.image_url && (
                  <img
                    src={ann.image_url.startsWith('http') || ann.image_url.startsWith('blob:') ? ann.image_url : `${API}${ann.image_url}`}
                    alt=""
                    className="announcement-hero-img"
                    onError={(e) => {
                      e.currentTarget.style.display = 'none';
                    }}
                  />
                )}

                <div style={{ padding: '18px 20px', display: 'flex', flexDirection: 'column', flex: 1 }}>
                  {/* Category & Priority Row */}
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, marginBottom: 10 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <span className={`badge ${isEmergency ? 'badge-red' : isImportant ? 'badge-amber' : 'badge-teal'}`}>
                        {ann.priority.toUpperCase()}
                      </span>
                      <span className="badge badge-muted" style={{ textTransform: 'capitalize' }}>
                        {ann.category.replace('_', ' ')}
                      </span>
                    </div>

                    {!ann.is_read && (
                      <button
                        type="button"
                        onClick={() => handleMarkAsRead(ann.id)}
                        className="btn btn-ghost btn-sm"
                        style={{ padding: '2px 8px', fontSize: 11 }}
                        title="Mark as read"
                      >
                        <CheckCircle size={12} /> Mark Read
                      </button>
                    )}
                  </div>

                  {/* Title & Content */}
                  <h3 style={{ fontSize: 16, fontWeight: 800, color: 'var(--text)', marginBottom: 8, lineHeight: 1.35 }}>
                    {ann.title}
                  </h3>
                  <p style={{ fontSize: 13, color: 'var(--text-dim)', lineHeight: 1.55, flex: 1, whiteSpace: 'pre-wrap' }}>
                    {ann.content}
                  </p>

                  {/* Footer: Author & Timestamp */}
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: 18, paddingTop: 12, borderTop: '1px solid var(--border)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <StaffAvatar
                        name={ann.author_name || 'Admin'}
                        avatarUrl={null}
                        size={28}
                        fontSize={11}
                      />
                      <div>
                        <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text)' }}>
                          {ann.author_name}
                        </div>
                        <div style={{ fontSize: 10.5, color: 'var(--text-muted)' }}>
                          {ann.created_at ? new Date(ann.created_at).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' }) : ''}
                        </div>
                      </div>
                    </div>

                    {/* Admin Action Buttons */}
                    {isAdmin && (
                      <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                        <button
                          type="button"
                          className="icon-btn"
                          style={{ width: 30, height: 30 }}
                          onClick={() => handleOpenEdit(ann)}
                          title="Edit announcement"
                        >
                          <Edit3 size={14} />
                        </button>
                        <button
                          type="button"
                          className="icon-btn"
                          style={{ width: 30, height: 30, color: 'var(--danger)' }}
                          onClick={() => handleDelete(ann.id)}
                          title="Delete announcement"
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Admin Create / Edit Modal */}
      {modalOpen && (
        <div
          className="modal-backdrop"
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 99999,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '24px 16px',
            background: 'rgba(3, 16, 24, 0.72)',
            backdropFilter: 'blur(4px)',
            overflowY: 'auto'
          }}
          onClick={(e) => {
            if (e.target === e.currentTarget) setModalOpen(false);
          }}
        >
          <div
            className="modal modal-md"
            style={{
              margin: 'auto',
              maxHeight: 'min(90vh, 620px)',
              overflowY: 'auto',
              display: 'flex',
              flexDirection: 'column',
              borderRadius: 16
            }}
          >
            <div className="modal-head" style={{ marginBottom: 14 }}>
              <h3 className="modal-title" style={{ fontSize: 17 }}>
                {editingAnn ? 'Edit Announcement' : 'Publish New Announcement'}
              </h3>
              <button type="button" className="icon-btn" onClick={() => setModalOpen(false)}>
                <X size={18} />
              </button>
            </div>

            {error && (
              <div className="alert alert-error" style={{ marginBottom: 14 }}>
                {error}
              </div>
            )}

            <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              <div className="form-field">
                <label className="form-label" style={{ marginBottom: 5 }}>Announcement Title</label>
                <input
                  type="text"
                  className="form-input"
                  placeholder="e.g. Hospital Maintenance Notice"
                  value={formTitle}
                  onChange={(e) => setFormTitle(e.target.value)}
                  required
                />
              </div>

              <div className="form-grid-2" style={{ gap: 12 }}>
                <div className="form-field">
                  <label className="form-label" style={{ marginBottom: 5 }}>Category</label>
                  <select
                    className="form-input"
                    value={formCategory}
                    onChange={(e) => setFormCategory(e.target.value)}
                  >
                    <option value="general">General Updates</option>
                    <option value="emergency">Emergency Notice</option>
                    <option value="policy">Policy Update</option>
                    <option value="training">Training & Education</option>
                    <option value="scheduling">Staff Scheduling</option>
                    <option value="events">Hospital Events</option>
                    <option value="new_staff">New Staff Welcome</option>
                  </select>
                </div>

                <div className="form-field">
                  <label className="form-label" style={{ marginBottom: 5 }}>Priority</label>
                  <select
                    className="form-input"
                    value={formPriority}
                    onChange={(e) => setFormPriority(e.target.value)}
                  >
                    <option value="normal">Normal</option>
                    <option value="important">Important</option>
                    <option value="emergency">Emergency</option>
                  </select>
                </div>
              </div>

              <div className="form-field">
                <label className="form-label" style={{ marginBottom: 5 }}>Cover Image (Optional)</label>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/png,image/jpeg,image/webp,image/gif"
                  style={{ display: 'none' }}
                  onChange={handleFileChange}
                />
                {imagePreview ? (
                  <div style={{
                    position: 'relative',
                    borderRadius: 10,
                    overflow: 'hidden',
                    border: '1px solid var(--border)',
                    background: 'var(--surface2)',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center'
                  }}>
                    <img
                      src={imagePreview}
                      alt="Cover Preview"
                      style={{
                        width: '100%',
                        maxHeight: 180,
                        objectFit: 'cover',
                        display: 'block'
                      }}
                    />
                    <div style={{
                      padding: '8px 12px',
                      width: '100%',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      background: 'rgba(0,0,0,0.4)',
                      backdropFilter: 'blur(4px)',
                      fontSize: 12,
                      color: 'var(--text)'
                    }}>
                      <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '65%' }}>
                        {selectedFile ? selectedFile.name : 'Cover image selected'}
                      </span>
                      <div style={{ display: 'flex', gap: 8 }}>
                        <button
                          type="button"
                          className="btn btn-ghost btn-xs"
                          onClick={() => fileInputRef.current?.click()}
                          style={{ padding: '2px 8px', fontSize: 11 }}
                        >
                          Change
                        </button>
                        <button
                          type="button"
                          className="btn btn-danger btn-xs"
                          onClick={handleRemoveImage}
                          style={{ padding: '2px 8px', fontSize: 11 }}
                        >
                          Remove
                        </button>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div
                    onClick={() => fileInputRef.current?.click()}
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={(e) => {
                      e.preventDefault();
                      const file = e.dataTransfer.files?.[0];
                      if (file) {
                        if (!file.type.startsWith('image/')) {
                          setError('Please drop a valid image file (PNG, JPG, WEBP, GIF).');
                          return;
                        }
                        if (file.size > 10 * 1024 * 1024) {
                          setError('Image exceeds 10MB limit.');
                          return;
                        }
                        setError('');
                        setSelectedFile(file);
                        setImagePreview(URL.createObjectURL(file));
                      }
                    }}
                    style={{
                      border: '2px dashed var(--border)',
                      borderRadius: 10,
                      padding: '20px 16px',
                      textAlign: 'center',
                      cursor: 'pointer',
                      background: 'var(--surface2)',
                      transition: 'border-color 0.15s ease',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      gap: 6
                    }}
                  >
                    <Upload size={22} style={{ color: 'var(--p)', opacity: 0.8 }} />
                    <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)' }}>
                      Click to upload cover image or drag and drop
                    </div>
                    <div style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>
                      Supports PNG, JPG, WEBP, GIF (up to 10MB)
                    </div>
                  </div>
                )}
              </div>

              <div className="form-field">
                <label className="form-label" style={{ marginBottom: 5 }}>Bulletin Description</label>
                <textarea
                  className="form-input"
                  rows={3}
                  placeholder="Detailed paragraph description of the notice or operational protocol..."
                  value={formContent}
                  onChange={(e) => setFormContent(e.target.value)}
                  required
                />
              </div>

              <div className="modal-footer" style={{ marginTop: 14, paddingTop: 12 }}>
                <button
                  type="button"
                  className="btn btn-ghost"
                  onClick={() => setModalOpen(false)}
                  disabled={submitting}
                >
                  Cancel
                </button>
                <button type="submit" className="btn btn-primary" disabled={submitting}>
                  {submitting ? 'Publishing...' : editingAnn ? 'Save Changes' : 'Publish Announcement'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
