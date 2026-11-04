"""
Recovera AI – Text-to-Speech (TTS) Proxy API
Provides natural, crystal-clear female voice audio synthesis for Tamil, Hindi, and English
using high-fidelity Google TTS streaming with in-memory caching.
"""
import io
import urllib.parse
from functools import lru_cache
import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

router = APIRouter(tags=["Text-to-Speech"])

# In-memory LRU cache for synthesized audio chunks
@lru_cache(maxsize=512)
def _fetch_tts_audio_cached(text: str, lang: str) -> bytes:
    """Fetches high-quality natural TTS audio for a short text chunk."""
    clean_text = text.strip()
    if not clean_text:
        return b""

    # Map language codes to Google TTS language identifiers
    lang_code = (lang or "en").lower().slice(0, 2) if hasattr(lang, "slice") else (lang or "en").lower()[:2]
    if lang_code == "ta":
        target_lang = "ta"
    elif lang_code == "hi":
        target_lang = "hi"
    else:
        target_lang = "en"

    # Truncate chunk to safe Google TTS query length (max 180 chars per request)
    # If text is longer, we slice into sentences/clauses and concatenate
    max_chunk_len = 160
    chunks = []
    if len(clean_text) <= max_chunk_len:
        chunks = [clean_text]
    else:
        words = clean_text.split()
        curr = []
        curr_len = 0
        for w in words:
            if curr_len + len(w) + 1 > max_chunk_len and curr:
                chunks.append(" ".join(curr))
                curr = [w]
                curr_len = len(w)
            else:
                curr.append(w)
                curr_len += len(w) + 1
        if curr:
            chunks.append(" ".join(curr))

    combined_bytes = bytearray()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Referer": "https://translate.google.com/",
    }

    with httpx.Client(timeout=8.0, follow_redirects=True) as client:
        for chunk in chunks:
            chunk_trimmed = chunk.strip()
            if not chunk_trimmed:
                continue
            params = {
                "ie": "UTF-8",
                "tl": target_lang,
                "client": "tw-ob",
                "q": chunk_trimmed,
            }
            resp = client.get("https://translate.google.com/translate_tts", params=params, headers=headers)
            if resp.status_code == 200 and resp.content:
                combined_bytes.extend(resp.content)

    return bytes(combined_bytes)


@router.api_route("/api/tts", methods=["GET", "HEAD"])
def get_text_to_speech(
    text: str = Query(..., min_length=1, max_length=1500, description="Text to synthesize"),
    lang: str = Query("en", description="Target language ('en', 'ta', 'hi')"),
):
    """
    Returns audio/mpeg stream of natural, understandable female speech for Tamil, Hindi, and English.
    """
    clean_text = text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    clean_lang = (lang or "en").lower()[:2]
    try:
        audio_data = _fetch_tts_audio_cached(clean_text, clean_lang)
        if not audio_data:
            raise HTTPException(status_code=502, detail="Unable to synthesize audio at this time.")

        return Response(
            content=audio_data,
            media_type="audio/mpeg",
            headers={
                "Cache-Control": "public, max-age=86400",
                "Content-Disposition": 'inline; filename="speech.mp3"',
                "Accept-Ranges": "bytes",
                "Content-Length": str(len(audio_data)),
                "Access-Control-Allow-Origin": "*",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"TTS service error: {str(e)}")
