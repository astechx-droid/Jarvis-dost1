"""
routers/tts.py — Text-to-Speech endpoints for JARVIS.

Endpoints:
  POST /tts/speak          — Text → MP3 audio (streaming)
  POST /tts/speak/bytes    — Text → MP3 audio (full buffer, good for short responses)
  GET  /tts/voices         — List all available voices

The /tts/speak endpoint streams audio chunks as they arrive from Edge TTS,
giving the lowest perceived latency — playback can start before the full
audio is generated.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field

from services.tts_service import (
    synthesize,
    synthesize_stream,
    get_available_voices,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tts", tags=["Text-to-Speech"])


# ── Request / Response schemas ────────────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to convert to speech.")
    language: str = Field(
        "english",
        description="Language of the text: 'english', 'hindi', 'hinglish', or ISO code ('en', 'hi')."
    )
    voice: Optional[str] = Field(
        None,
        description=(
            "Override voice name (e.g. 'en-GB-RyanNeural', 'hi-IN-SwaraNeural'). "
            "If omitted, best voice for the language is selected automatically."
        )
    )
    rate: Optional[str] = Field(
        None,
        description="Speaking rate offset: '+10%', '-5%', etc. Default is language-tuned."
    )
    pitch: Optional[str] = Field(
        None,
        description="Pitch offset in Hz: '+5Hz', '-10Hz'. Default is '-5Hz' (JARVIS tone)."
    )
    stream: bool = Field(
        True,
        description="Stream audio chunks as they arrive (lower latency). Set false to get full buffer."
    )


# ── POST /tts/speak — primary endpoint (streaming) ───────────────────────────

@router.post(
    "/speak",
    summary="Convert text to speech (streaming MP3)",
    description=(
        "Converts text to natural speech using Microsoft Edge TTS neural voices. "
        "Streams MP3 audio chunks for low-latency playback. "
        "Supports Hindi, English, and Hinglish."
    ),
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"audio/mpeg": {}},
            "description": "MP3 audio stream of the synthesized speech.",
        }
    },
)
async def speak_stream(req: TTSRequest):
    """
    Primary TTS endpoint — streams MP3 audio directly to the client.

    The client receives audio chunks as Edge TTS generates them,
    so playback can begin before the full response is ready.
    """
    logger.info(
        "TTS /speak | lang=%s | voice=%s | stream=%s | chars=%d",
        req.language, req.voice or "auto", req.stream, len(req.text)
    )

    if req.stream:
        # ── Streaming mode — lower latency ──────────────────────────────────
        async def audio_generator():
            try:
                async for chunk in synthesize_stream(
                    text=req.text,
                    language=req.language,
                    voice=req.voice,
                    rate=req.rate,
                    pitch=req.pitch,
                ):
                    yield chunk
            except Exception as e:
                logger.error("TTS stream error: %s", e)
                # Cannot send HTTP error mid-stream; log and stop
                return

        return StreamingResponse(
            audio_generator(),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "inline; filename=jarvis_speech.mp3",
                "Cache-Control": "no-store",
                "X-Voice": req.voice or "auto",
                "X-Language": req.language,
            },
        )
    else:
        # ── Buffered mode — full MP3 at once ────────────────────────────────
        try:
            audio_bytes = await synthesize(
                text=req.text,
                language=req.language,
                voice=req.voice,
                rate=req.rate,
                pitch=req.pitch,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            logger.error("TTS error: %s", e)
            raise HTTPException(status_code=502, detail=f"TTS synthesis failed: {e}")

        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "inline; filename=jarvis_speech.mp3",
                "Content-Length": str(len(audio_bytes)),
                "Cache-Control": "no-store",
                "X-Voice": req.voice or "auto",
                "X-Language": req.language,
            },
        )


# ── GET /tts/voices ───────────────────────────────────────────────────────────

@router.get(
    "/voices",
    summary="List available TTS voices",
    description="Returns all available Edge TTS neural voices with metadata.",
)
async def list_voices():
    """Return the complete voice catalogue with gender, accent, and style info."""
    return {
        "provider": "Microsoft Edge TTS (Neural)",
        "voices": get_available_voices(),
        "tip": (
            "For Hinglish, 'hi-IN-SwaraNeural' sounds the most natural "
            "for mixed Hindi-English speech. "
            "For a JARVIS-style English voice, use 'en-GB-RyanNeural'."
        ),
    }
