import React, { useState, useEffect } from 'react';
import { Search, X, ChevronUp, ChevronDown } from 'lucide-react';
import { formatMessageDateSeparator, formatMessageTime, API } from './staffChatUtils';

export function StaffSearchDrawer({
  conversationId,
  token,
  onClose,
  onSelectResult,
}) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!query.trim()) {
      setResults([]);
      return;
    }

    const timer = setTimeout(async () => {
      setLoading(true);
      try {
        const res = await fetch(`${API}/api/staff-chat/conversations/${conversationId}/search?q=${encodeURIComponent(query)}`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        if (res.ok) {
          const data = await res.json();
          setResults(data);
          setCurrentIndex(0);
        }
      } catch (err) {
        console.error('Search error:', err);
      } finally {
        setLoading(false);
      }
    }, 250);

    return () => clearTimeout(timer);
  }, [query, conversationId, token]);

  const handleNext = () => {
    if (results.length === 0) return;
    const next = (currentIndex + 1) % results.length;
    setCurrentIndex(next);
    onSelectResult(results[next].id);
  };

  const handlePrev = () => {
    if (results.length === 0) return;
    const prev = (currentIndex - 1 + results.length) % results.length;
    setCurrentIndex(prev);
    onSelectResult(results[prev].id);
  };

  const highlightMatches = (text, term) => {
    if (!term || !text) return text;
    const parts = text.split(new RegExp(`(${term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi'));
    return parts.map((part, i) =>
      part.toLowerCase() === term.toLowerCase() ? (
        <mark key={i} style={{ background: '#fef08a', color: '#854d0e', padding: '0 2px', borderRadius: 2 }}>
          {part}
        </mark>
      ) : (
        part
      )
    );
  };

  return (
    <div className="staff-search-drawer">
      {/* Top search bar with count & up/down */}
      <div style={{ padding: '14px 16px', borderBottom: '1px solid var(--border)', display: 'flex', flexDirection: 'column', gap: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--text)' }}>Search in conversation</span>
          <button type="button" className="staff-composer-btn" onClick={onClose}>
            <X size={18} />
          </button>
        </div>

        <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
          <Search size={15} style={{ position: 'absolute', left: 12, color: 'var(--text-muted)' }} />
          <input
            type="text"
            className="form-input"
            style={{ paddingLeft: 34, paddingRight: 74, fontSize: 13, height: 36 }}
            placeholder="Search this chat..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            autoFocus
          />

          {query && (
            <div style={{ position: 'absolute', right: 8, display: 'flex', alignItems: 'center', gap: 4 }}>
              <button
                type="button"
                onClick={handlePrev}
                disabled={results.length === 0}
                style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', padding: 2 }}
                title="Previous match"
              >
                <ChevronUp size={16} />
              </button>
              <button
                type="button"
                onClick={handleNext}
                disabled={results.length === 0}
                style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', padding: 2 }}
                title="Next match"
              >
                <ChevronDown size={16} />
              </button>
            </div>
          )}
        </div>

        {query && (
          <div style={{ fontSize: 11.5, color: 'var(--text-muted)', fontWeight: 600 }}>
            {loading ? 'Searching...' : `${results.length} result${results.length === 1 ? '' : 's'}`}
          </div>
        )}
      </div>

      {/* Results List */}
      <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column' }}>
        {results.map((res, i) => (
          <div
            key={res.id}
            onClick={() => {
              setCurrentIndex(i);
              onSelectResult(res.id);
            }}
            style={{
              padding: '12px 16px',
              borderBottom: '1px solid rgba(255,255,255,0.05)',
              cursor: 'pointer',
              background: currentIndex === i ? 'rgba(0,229,192,0.1)' : 'transparent',
              transition: 'background 0.12s',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
              <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--text)' }}>
                {res.sender_name}
              </span>
              <span style={{ fontSize: 10.5, color: 'var(--text-muted)' }}>
                {formatMessageDateSeparator(res.created_at)} {formatMessageTime(res.created_at)}
              </span>
            </div>
            <div style={{ fontSize: 12.5, color: 'var(--text-dim)', lineHeight: 1.4 }}>
              {highlightMatches(res.content, query)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
