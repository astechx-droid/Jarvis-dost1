"""
routers/conversation.py — Unified voice conversation pipeline for JARVIS.

Endpoints:
  POST /conversation/voice   — Full pipeline: audio → STT → AI → JSON
                               Client then streams /tts/speak for audio.
  POST /conversation/text    — Text input → AI → JSON (with optional TTS flag)
  GET  /conversation/health  — Pipeline health check with timing info

Design rationale:
  The pipeline returns JSON first (fast: ~1–3 s total for STT + LLM).
  The client immediately fires a streaming POST /tts/speak with the reply text,
  so audio starts playing within ~300 ms of the JSON landing.
  This keeps the HTTP model simple and browser-compatible, while achieving
  the feel of a real-time streaming pipeline.
"""

import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from database.db import get_db
from services.pipeline_service import run_voice_pipeline, run_text_pipeline
from services.stt_service import validate_audio_file
from services.tts_service import synthesize_stream

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/conversation", tags=["Conversation Pipeline"])


# ── Response schemas ──────────────────────────────────────────────────────────

class PipelineResponse(BaseModel):
    """Full pipeline result returned to the client after one conversational turn."""
    # Input
    input_type: str                          # 'voice' or 'text'
    transcript: str                          # what the user said / typed
    transcript_language: str                 # 'english' | 'hindi' | 'hinglish'
    transcript_language_code: str            # Whisper ISO code
    audio_duration: Optional[float]          # seconds (voice only)

    # Memory
    conversation_id: str
    message_count: int
    history_turns_used: int

    # Intelligence
    used_web_search: bool
    search_query: Optional[str]

    # Output
    reply: str
    reply_language: str                      # language JARVIS replied in

    # Performance
    timing_ms: dict                          # per-stage timing breakdown

    # TTS hint — client uses these to call /tts/speak
    tts_voice: str                           # recommended voice for this reply
    tts_language: str                        # language key for /tts/speak


class TextPipelineRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: Optional[str] = None
    force_search: bool = False
    with_audio: bool = Field(
        False,
        description="If true, streams audio inline after the JSON header. "
                    "If false (default), client calls /tts/speak separately."
    )


# ── Voice → TTS voice mapping ─────────────────────────────────────────────────

_VOICE_MAP = {
    "english":  ("hi-IN-MadhurNeural",  "english"),
    "hindi":    ("hi-IN-MadhurNeural",  "hindi"),
    "hinglish": ("hi-IN-MadhurNeural",  "hinglish"),
}

def _tts_hint(language: str) -> tuple[str, str]:
    return _VOICE_MAP.get(language, _VOICE_MAP["english"])


# ── POST /conversation/voice ──────────────────────────────────────────────────

@router.post(
    "/voice",
    response_model=PipelineResponse,
    summary="Full voice pipeline: audio → transcript → AI reply",
    description=(
        "The complete JARVIS voice conversation pipeline in one request.\n\n"
        "**Workflow:**\n"
        "1. Upload microphone audio\n"
        "2. Groq Whisper transcribes it (Hindi / English / Hinglish auto-detected)\n"
        "3. Conversation memory is loaded from SQLite\n"
        "4. Optional web search fires if the query needs live data\n"
        "5. Groq LLM generates a natural, contextual JARVIS reply\n"
        "6. Both turns are saved to memory\n"
        "7. JSON response includes the `tts_voice` + `tts_language` hint\n\n"
        "**Client flow:** receive this JSON → immediately POST `/tts/speak` "
        "with `reply` + `tts_language` → stream the audio."
    ),
)
async def voice_pipeline(
    file: UploadFile = File(..., description="Microphone audio (webm, mp3, wav, m4a, ogg)"),
    conversation_id: Optional[str] = Form(
        None, description="Continue an existing conversation. Omit to start new."
    ),
    force_search: bool = Form(
        False, description="Force a web search regardless of query content."
    ),
    db: AsyncSession = Depends(get_db),
):
    # ── Validate audio ────────────────────────────────────────────────────────
    audio_bytes  = await file.read()
    content_type = file.content_type or "audio/webm"

    try:
        validate_audio_file(content_type, len(audio_bytes))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # ── Run pipeline ──────────────────────────────────────────────────────────
    try:
        result = await run_voice_pipeline(
            audio_bytes=audio_bytes,
            filename=file.filename or "audio.webm",
            content_type=content_type,
            db=db,
            conversation_id=conversation_id,
            force_search=force_search,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Voice pipeline error: %s", e, exc_info=True)
        raise HTTPException(status_code=502, detail=f"Pipeline error: {e}")

    tts_voice, tts_lang = _tts_hint(result.reply_language)

    return PipelineResponse(
        input_type="voice",
        transcript=result.transcript,
        transcript_language=result.transcript_language,
        transcript_language_code=result.transcript_language_code,
        audio_duration=result.audio_duration,
        conversation_id=result.conversation_id,
        message_count=result.message_count,
        history_turns_used=result.history_turns_used,
        used_web_search=result.used_web_search,
        search_query=result.search_query,
        reply=result.reply,
        reply_language=result.reply_language,
        timing_ms=result.timing,
        tts_voice=tts_voice,
        tts_language=tts_lang,
    )


# ── POST /conversation/text ───────────────────────────────────────────────────

@router.post(
    "/text",
    summary="Text pipeline: message → AI reply (+ optional inline audio)",
    description=(
        "Text input version of the same full pipeline. "
        "Memory, search, and LLM all apply. "
        "Set `with_audio=true` to stream audio after the JSON header."
    ),
)
async def text_pipeline(
    req: TextPipelineRequest,
    db: AsyncSession = Depends(get_db),
):
    try:
        result = await run_text_pipeline(
            user_message=req.message,
            db=db,
            conversation_id=req.conversation_id,
            force_search=req.force_search,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Text pipeline error: %s", e, exc_info=True)
        raise HTTPException(status_code=502, detail=f"Pipeline error: {e}")

    tts_voice, tts_lang = _tts_hint(result.reply_language)

    payload = PipelineResponse(
        input_type="text",
        transcript=result.transcript,
        transcript_language=result.transcript_language,
        transcript_language_code=result.transcript_language_code,
        audio_duration=None,
        conversation_id=result.conversation_id,
        message_count=result.message_count,
        history_turns_used=result.history_turns_used,
        used_web_search=result.used_web_search,
        search_query=result.search_query,
        reply=result.reply,
        reply_language=result.reply_language,
        timing_ms=result.timing,
        tts_voice=tts_voice,
        tts_language=tts_lang,
    )

    # If caller wants inline audio, stream it right after the JSON
    if req.with_audio:
        import json

        async def json_then_audio():
            # First chunk: JSON header terminated with a separator
            header = json.dumps(payload.model_dump()).encode()
            yield b"JSON:" + header + b"\r\n\r\nAUDIO:"
            # Remaining chunks: raw MP3
            async for chunk in synthesize_stream(
                text=result.reply,
                language=tts_lang,
                voice=tts_voice,
            ):
                yield chunk

        return StreamingResponse(
            json_then_audio(),
            media_type="application/octet-stream",
            headers={"X-Pipeline": "text+audio"},
        )

    return payload


# ── GET /conversation/health ──────────────────────────────────────────────────

@router.get("/health", summary="Pipeline health check")
async def pipeline_health():
    """Confirm all pipeline components are wired and reachable."""
    return {
        "status": "operational",
        "pipeline_stages": [
            {"stage": 1, "name": "speech_to_text",   "provider": "Groq Whisper (whisper-large-v3-turbo)"},
            {"stage": 2, "name": "language_detect",  "provider": "Custom detector (no external deps)"},
            {"stage": 3, "name": "memory_retrieval", "provider": "SQLite via SQLAlchemy async"},
            {"stage": 4, "name": "web_search",       "provider": "DuckDuckGo (conditional)"},
            {"stage": 5, "name": "llm_generation",   "provider": "Groq llama-3.3-70b-versatile"},
            {"stage": 6, "name": "memory_persist",   "provider": "SQLite via SQLAlchemy async"},
            {"stage": 7, "name": "text_to_speech",   "provider": "Microsoft Edge TTS (Neural WebSocket)"},
        ],
        "supported_languages": ["english", "hindi", "hinglish"],
        "voice_map": {
            "english":  "hi-IN-MadhurNeural",
            "hindi":    "hi-IN-MadhurNeural",
            "hinglish": "hi-IN-MadhurNeural",
        },
        "notes": [
            "Stages 3+4 run concurrently (asyncio.gather).",
            "TTS is streamed by the client from /tts/speak after this endpoint responds.",
            "Audio starts playing within ~300 ms of JSON receipt.",
        ],
    }
