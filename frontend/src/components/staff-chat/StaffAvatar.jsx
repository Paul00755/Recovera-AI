import React, { useState } from 'react';

/**
 * Robust StaffAvatar with automated fallback to styled initials if image fails or is unavailable.
 * Never renders broken image alt text over status badges.
 */
export function StaffAvatar({
  name = 'Staff',
  avatarUrl = null,
  status = null,
  size = 42,
  fontSize = 14,
  style = {},
  className = '',
}) {
  const [imgError, setImgError] = useState(false);

  // Compute initials cleanly
  const trimmed = (name || '').trim();
  const initial = trimmed ? trimmed.charAt(0).toUpperCase() : '?';

  return (
    <div
      className={`staff-avatar-wrap ${className}`}
      style={{
        width: size,
        height: size,
        minWidth: size,
        minHeight: size,
        flexShrink: 0,
        position: 'relative',
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: '50%',
        ...style,
      }}
    >
      {avatarUrl && !imgError ? (
        <img
          src={avatarUrl}
          alt=""
          className="staff-avatar-img"
          onError={() => setImgError(true)}
          style={{
            width: '100%',
            height: '100%',
            borderRadius: '50%',
            objectFit: 'cover',
            display: 'block',
          }}
        />
      ) : (
        <div
          className="staff-avatar-initials"
          style={{
            width: '100%',
            height: '100%',
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontWeight: 800,
            fontSize: fontSize,
            background: 'linear-gradient(135deg, #0d3b38, #1e3a6a)',
            color: '#ffffff',
            userSelect: 'none',
          }}
        >
          {initial}
        </div>
      )}
      {status && <span className={`staff-status-badge ${status}`} />}
    </div>
  );
}

export default StaffAvatar;
