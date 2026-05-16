"""
services/tts_service.py — Text-to-Speech service for JARVIS using Microsoft Edge TTS.

Clean, high-speed configuration using hi-IN-MadhurNeural.
"""

import asyncio
import logging
import os
import re
import subprocess
from typing import Optional, AsyncGenerator

import edge_tts

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_MALE_VOICE = "hi-IN-MadhurNeural"
DEFAULT_INDIAN_VOICE = DEFAULT_MALE_VOICE

# High-speed JARVIS (1.25x)
DEFAULT_RATE = "+25%"
DEFAULT_PITCH = "+0Hz"

_VOICE_MAP = {
    "english":  DEFAULT_MALE_VOICE,
    "hindi":    DEFAULT_MALE_VOICE,
    "hinglish": DEFAULT_MALE_VOICE,
}

# ── Public API ────────────────────────────────────────────────────────────────

async def synthesize(
    text: str,
    language: str = "english",
    voice: Optional[str] = None,
    rate: Optional[str] = None,
    pitch: Optional[str] = None,
) -> bytes:
    """Synthesise text to MP3 bytes using Edge TTS."""
    clean_text = _clean_text(text)
    selected_voice = voice or DEFAULT_MALE_VOICE
    final_rate = rate or DEFAULT_RATE
    final_pitch = pitch or DEFAULT_PITCH
    
    communicate = edge_tts.Communicate(clean_text, selected_voice, rate=final_rate, pitch=final_pitch)
    
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
            
    return audio_data

async def synthesize_stream(
    text: str,
    language: str = "english",
    voice: Optional[str] = None,
    rate: Optional[str] = None,
    pitch: Optional[str] = None,
) -> AsyncGenerator[bytes, None]:
    """Async generator yielding MP3 chunks."""
    clean_text = _clean_text(text)
    selected_voice = voice or DEFAULT_MALE_VOICE
    final_rate = rate or DEFAULT_RATE
    final_pitch = pitch or DEFAULT_PITCH
    
    communicate = edge_tts.Communicate(clean_text, selected_voice, rate=final_rate, pitch=final_pitch)
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            yield chunk["data"]

# ── Reusable speak() function ────────────────────────────────────────────────

def speak(text: str, voice: str = DEFAULT_MALE_VOICE):
    """Clean, high-speed speak function."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        asyncio.create_task(_speak_async(text, voice))
    else:
        loop.run_until_complete(_speak_async(text, voice))

async def _speak_async(text: str, voice: str):
    """Internal async implementation."""
    output_file = "voice.mp3"
    clean_text = _clean_text(text)
    communicate = edge_tts.Communicate(clean_text, voice, rate=DEFAULT_RATE, pitch=DEFAULT_PITCH)
    await communicate.save(output_file)
    
    if os.getenv("CODESPACES") == "true":
        print(f"\n[JARVIS]: {text}")
    else:
        try:
            subprocess.Popen(["mpv", "--no-video", output_file], 
                             stdout=subprocess.DEVNULL, stderr=subprocess.STNULL)
        except Exception:
            print(f"\n[JARVIS]: {text}")

def _clean_text(text: str) -> str:
    """Minimal cleaning for Edge-TTS."""
    text = re.sub(r'[\*\_\`\#\[\]\(\)\<\>]', '', text)
    text = re.sub(r'https?://\S+', 'link', text)
    return text.strip()
