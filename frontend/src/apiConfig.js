export function getApiBase() {
  let raw = import.meta.env.VITE_API_URL;
  if (raw && !raw.includes('.') && !raw.includes('localhost')) {
    raw = `${raw}.onrender.com`;
  }
  if (raw) {
    return raw.startsWith('http') ? raw : `https://${raw}`;
  }
  if (typeof window !== 'undefined' && window.location.origin.includes('localhost')) {
    return 'http://localhost:8000';
  }
  return 'https://recovera-ai-backend.onrender.com';
}

export const API_BASE = getApiBase();
