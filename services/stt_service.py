"""
services/stt_service.py — Speech-to-Text service for JARVIS.

Uses Groq's Whisper API (whisper-large-v3-turbo) for ultra-fast transcription.
Supports Hindi, English, Hinglish, and auto-language detection.

Why Groq Whisper:
  - No local model download (runs on Groq's infrastructure)
  - Extremely fast (~300x real-time speed)
  - Supports 99 languages including Hindi
  - Uses the same API key already configured
"""

import io
import logging
import tempfile
import os
from typing import Optional

import httpx
from config import settings

logger = logging.getLogger(__name__)

# ── Supported audio MIME types → file extensions ──────────────────────────────
SUPPORTED_MIME_TYPES = {
    "audio/mpeg":       "mp3",
    "audio/mp3":        "mp3",
    "audio/mp4":        "mp4",
    "audio/x-m4a":     "m4a",
    "audio/m4a":        "m4a",
    "audio/wav":        "wav",
    "audio/x-wav":      "wav",
    "audio/wave":       "wav",
    "audio/webm":       "webm",
    "video/webm":       "webm",   # browser MediaRecorder often sends this
    "audio/ogg":        "ogg",
    "audio/flac":       "flac",
}

# Languages JARVIS supports for forced transcription
LANGUAGE_CODES = {
    "hindi":    "hi",
    "english":  "en",
    "hinglish": None,   # None = auto-detect (Whisper handles mixed-language well)
    "auto":     None,
}

# ── Groq Whisper API endpoint ─────────────────────────────────────────────────
GROQ_AUDIO_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


async def transcribe_audio(
    audio_bytes: bytes,
    filename: str,
    content_type: str,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> dict:
    """
    Transcribe audio using Groq Whisper API.

    Args:
        audio_bytes:  Raw audio file bytes.
        filename:     Original filename (used to hint format).
        content_type: MIME type of the audio (e.g. 'audio/webm').
        language:     Optional ISO-639-1 language code ('en', 'hi').
                      Pass None to auto-detect.
        prompt:       Optional context prompt to guide transcription
                      (e.g. "Mr Aryan is speaking in Hinglish").

    Returns:
        dict with keys:
          - text:      The transcribed text
          - language:  Detected/forced language code
          - duration:  Audio duration in seconds (if available)
          - segments:  Detailed word segments (if verbose)
    """
    if not audio_bytes:
        raise ValueError("Empty audio data received.")

    if len(audio_bytes) > 25 * 1024 * 1024:  # 25 MB Groq limit
        raise ValueError("Audio file too large. Maximum size is 25 MB.")

    # Determine file extension from MIME type or filename
    ext = _get_extension(content_type, filename)

    # Default prompt to help Whisper with Hinglish context
    if prompt is None:
        prompt = (
            "Mr Aryan is speaking. He may speak in Hindi, English, or Hinglish "
            "(a mix of Hindi and English). Transcribe accurately."
        )

    logger.info(
        "Transcription request | size=%d bytes | ext=%s | lang=%s",
        len(audio_bytes), ext, language or "auto"
    )

    # Build multipart form data for Groq API
    files = {
        "file": (f"audio.{ext}", io.BytesIO(audio_bytes), content_type or "audio/webm"),
    }
    data = {
        "model":             "whisper-large-v3-turbo",   # fastest high-quality Whisper
        "response_format":   "verbose_json",              # get language + duration info
        "prompt":            prompt,
    }
    if language:
        data["language"] = language  # ISO-639-1 code

    # Call Groq Whisper API
    import ssl
    ssl_ctx = ssl.create_default_context()

    async with httpx.AsyncClient(verify=ssl_ctx, timeout=60.0) as client:
        response = await client.post(
            GROQ_AUDIO_URL,
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            files=files,
            data=data,
        )

    if response.status_code != 200:
        error_body = response.text
        logger.error("Groq Whisper API error | status=%d | body=%s", response.status_code, error_body)
        raise ValueError(f"Transcription failed (HTTP {response.status_code}): {error_body}")

    result = response.json()

    text      = result.get("text", "").strip()
    detected  = result.get("language", language or "unknown")
    duration  = result.get("duration", None)
    segments  = result.get("segments", [])

    logger.info(
        "Transcription complete | lang=%s | duration=%.1fs | text_len=%d",
        detected, duration or 0, len(text)
    )

    return {
        "text":     text,
        "language": detected,
        "duration": duration,
        "segments": segments,
    }


async def transcribe_and_detect(
    audio_bytes: bytes,
    filename: str,
    content_type: str,
) -> dict:
    """
    Transcribe audio with automatic language detection.
    Adds JARVIS-specific language classification on top of Whisper's detection.

    Returns:
        dict with keys: text, language, language_code, duration
    """
    result = await transcribe_audio(
        audio_bytes=audio_bytes,
        filename=filename,
        content_type=content_type,
        language=None,  # auto-detect
    )

    text = result["text"]
    detected_code = result["language"]

    # Map Whisper's language code to JARVIS language label
    jarvis_lang = _map_language(detected_code, text)

    return {
        "text":          text,
        "language":      jarvis_lang,
        "language_code": detected_code,
        "duration":      result["duration"],
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_extension(content_type: str, filename: str) -> str:
    """Determine audio file extension from MIME type or filename."""
    # Try MIME type first
    ext = SUPPORTED_MIME_TYPES.get(content_type.lower().split(";")[0].strip())
    if ext:
        return ext

    # Fall back to filename extension
    if filename and "." in filename:
        return filename.rsplit(".", 1)[-1].lower()

    # Default to webm (common from browser MediaRecorder)
    return "webm"


def _map_language(whisper_code: str, text: str) -> str:
    """
    Map Whisper's ISO-639-1 code to JARVIS language label.
    Also detects Hinglish (Hindi words in Latin script).
    """
    if not whisper_code:
        return "hinglish"

    code = whisper_code.lower()

    if code == "hi":
        # Check if it's actually Hinglish (Latin script in primarily Hindi output)
        import re
        latin_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = max(len(text), 1)
        if latin_chars / total_chars > 0.3:
            return "hinglish"
        return "hindi"
    elif code == "en":
        # Check for Hinglish markers in English-detected text
        hinglish_markers = {
            "kya", "hai", "hain", "aap", "main", "nahi", "bhi", "aur",
            "toh", "agar", "lekin", "bahut", "bohot", "yaar", "bhai",
            "accha", "theek", "matlab", "samajh", "bilkul", "zaroor",
        }
        words = set(text.lower().split())
        if words & hinglish_markers:
            return "hinglish"
        return "english"
    else:
        return code  # pass through other languages as-is


def validate_audio_file(content_type: str, size_bytes: int) -> None:
    """
    Validate audio file before processing.
    Raises ValueError with a descriptive message if invalid.
    """
    # Check MIME type
    normalized = content_type.lower().split(";")[0].strip()
    if normalized not in SUPPORTED_MIME_TYPES and not normalized.startswith("audio/"):
        raise ValueError(
            f"Unsupported file type: '{content_type}'. "
            f"Supported formats: mp3, wav, webm, m4a, ogg, flac, mp4."
        )

    # Check size (25 MB Groq limit)
    if size_bytes > 25 * 1024 * 1024:
        raise ValueError("Audio file too large. Maximum size is 25 MB.")

    # Warn about very small files
    if size_bytes < 100:
        raise ValueError("Audio file is too small. Please provide a valid recording.")
