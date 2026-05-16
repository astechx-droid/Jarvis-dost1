"""
services/pipeline_service.py — Unified real-time voice conversation pipeline for JARVIS.

Stages (in order, with timing):
  1. STT         — Groq Whisper transcribes audio
  2. Language    — Detect Hindi / English / Hinglish from transcript
  3. Memory      — Load relevant conversation history from SQLite
  4. Search      — Optional DuckDuckGo web search (parallel with memory load)
  5. LLM         — Groq llama-3.3-70b generates JARVIS reply
  6. Persist     — Save both turns to memory (non-blocking)
  7. TTS-ready   — Return text; caller streams /tts/speak for audio

Latency strategy:
  • Memory load and search trigger fire concurrently (asyncio.gather)
  • STT and LLM calls are on Groq's fast inference — typically <1 s each
  • TTS is streamed by the client immediately after JSON is returned,
    so audio starts playing within ~0.3 s of the JSON response landing
  • DB writes are fire-and-forget (awaited at end, don't block the response)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict

from sqlalchemy.ext.asyncio import AsyncSession

from services.stt_service import transcribe_and_detect
from services.memory_service import (
    get_or_create_conversation,
    get_conversation_history,
    save_message,
    get_message_count,
)
from services.search_service import web_search, extract_search_query
from services.groq_service import chat_completion
from core.personality import detect_language, should_search_web
from core.context import build_messages
from config import settings

logger = logging.getLogger(__name__)


# ── Pipeline result dataclass ─────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Structured output of one full voice conversation turn."""
    # STT
    transcript: str
    transcript_language: str          # 'english' | 'hindi' | 'hinglish'
    transcript_language_code: str     # ISO code from Whisper ('en', 'hi', …)
    audio_duration: Optional[float]

    # Memory
    conversation_id: str
    message_count: int
    history_turns_used: int

    # Search
    used_web_search: bool
    search_query: Optional[str]

    # LLM
    reply: str
    reply_language: str               # detected language of the reply

    # Timings (milliseconds)
    timing: Dict[str, int] = field(default_factory=dict)


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def run_voice_pipeline(
    audio_bytes: bytes,
    filename: str,
    content_type: str,
    db: AsyncSession,
    conversation_id: Optional[str] = None,
    force_search: bool = False,
    auto_search: bool = True,
) -> PipelineResult:
    """
    Execute the full voice → text → AI → (TTS-ready) pipeline.

    Args:
        audio_bytes:     Raw audio data from the microphone.
        filename:        Original filename hint (for format detection).
        content_type:    MIME type of the audio.
        db:              Async SQLAlchemy session.
        conversation_id: Existing conversation to continue, or None for new.
        force_search:    Always trigger web search, regardless of content.
        auto_search:     Allow automatic search based on content heuristics.

    Returns:
        PipelineResult with all pipeline outputs and timing data.
    """
    t_start = time.monotonic()
    timing: Dict[str, int] = {}

    # ── STAGE 1: Speech-to-Text ───────────────────────────────────────────────
    logger.info("Pipeline | STT start | size=%d bytes", len(audio_bytes))
    t0 = time.monotonic()

    stt_result = await transcribe_and_detect(
        audio_bytes=audio_bytes,
        filename=filename,
        content_type=content_type,
    )

    timing["stt_ms"] = int((time.monotonic() - t0) * 1000)
    transcript = stt_result["text"].strip()
    logger.info(
        "Pipeline | STT done | %dms | lang=%s | text=%r",
        timing["stt_ms"], stt_result["language"], transcript[:60]
    )

    if not transcript:
        raise ValueError(
            "No speech detected in the audio. "
            "Please speak clearly and try again, Mr Aryan."
        )

    # ── STAGE 2: Language detection ───────────────────────────────────────────
    # Use our custom detector for final language classification
    # (Whisper gives broad ISO code; we refine to 'hinglish' etc.)
    detected_lang = stt_result["language"]   # already refined by stt_service
    timing["lang_ms"] = 0                    # done inside STT stage

    # ── STAGE 3 + 4: Memory load & optional web search — run concurrently ─────
    t0 = time.monotonic()

    conv = await get_or_create_conversation(db, conversation_id)

    # Decide on search
    do_search = force_search or (auto_search and should_search_web(transcript))
    search_query_str: Optional[str] = None

    if do_search:
        search_query_str = extract_search_query(transcript)
        history, search_results = await asyncio.gather(
            get_conversation_history(db, conv.id, max_turns=settings.max_history_turns),
            web_search(search_query_str),
        )
    else:
        history = await get_conversation_history(db, conv.id, max_turns=settings.max_history_turns)
        search_results: List[Dict] = []

    timing["memory_search_ms"] = int((time.monotonic() - t0) * 1000)
    logger.info(
        "Pipeline | Memory+Search done | %dms | history=%d msgs | search=%s",
        timing["memory_search_ms"], len(history), do_search
    )

    # ── STAGE 5: Build context and call Groq LLM ─────────────────────────────
    t0 = time.monotonic()

    messages = build_messages(
        history=history,
        user_message=transcript,
        search_results=search_results or None,
        search_query=search_query_str,
    )

    reply = await chat_completion(messages)

    timing["llm_ms"] = int((time.monotonic() - t0) * 1000)
    logger.info(
        "Pipeline | LLM done | %dms | reply_len=%d",
        timing["llm_ms"], len(reply)
    )

    # Detect the language JARVIS replied in (for TTS voice selection)
    reply_language = detect_language(reply)

    # ── STAGE 6: Persist both turns to memory (sequential — same DB session) ──
    t0 = time.monotonic()

    await save_message(db, conv.id, "user", transcript)
    await save_message(
        db, conv.id, "assistant", reply,
        used_search=bool(search_results),
        search_query=search_query_str,
    )

    timing["persist_ms"] = int((time.monotonic() - t0) * 1000)

    # ── Total ─────────────────────────────────────────────────────────────────
    timing["total_ms"] = int((time.monotonic() - t_start) * 1000)

    msg_count = await get_message_count(db, conv.id)

    logger.info(
        "Pipeline | COMPLETE | total=%dms (stt=%d llm=%d mem=%d persist=%d) | conv=%s",
        timing["total_ms"], timing["stt_ms"], timing["llm_ms"],
        timing["memory_search_ms"], timing["persist_ms"], conv.id
    )

    return PipelineResult(
        transcript=transcript,
        transcript_language=stt_result["language"],
        transcript_language_code=stt_result["language_code"],
        audio_duration=stt_result["duration"],
        conversation_id=conv.id,
        message_count=msg_count,
        history_turns_used=len(history) // 2,
        used_web_search=bool(search_results),
        search_query=search_query_str,
        reply=reply,
        reply_language=reply_language,
        timing=timing,
    )


async def run_text_pipeline(
    user_message: str,
    db: AsyncSession,
    conversation_id: Optional[str] = None,
    force_search: bool = False,
    auto_search: bool = True,
) -> PipelineResult:
    """
    Same pipeline but starting from text (no STT stage).
    Used for the combined text-input → TTS-output flow.
    """
    t_start = time.monotonic()
    timing: Dict[str, int] = {}

    transcript = user_message.strip()
    if not transcript:
        raise ValueError("Empty message received.")

    detected_lang = detect_language(transcript)
    timing["stt_ms"] = 0   # no STT for text input

    t0 = time.monotonic()
    conv = await get_or_create_conversation(db, conversation_id)

    do_search = force_search or (auto_search and should_search_web(transcript))
    search_query_str: Optional[str] = None

    if do_search:
        search_query_str = extract_search_query(transcript)
        history, search_results = await asyncio.gather(
            get_conversation_history(db, conv.id, max_turns=settings.max_history_turns),
            web_search(search_query_str),
        )
    else:
        history = await get_conversation_history(db, conv.id, max_turns=settings.max_history_turns)
        search_results: List[Dict] = []

    timing["memory_search_ms"] = int((time.monotonic() - t0) * 1000)

    t0 = time.monotonic()
    messages = build_messages(
        history=history,
        user_message=transcript,
        search_results=search_results or None,
        search_query=search_query_str,
    )
    reply = await chat_completion(messages)
    timing["llm_ms"] = int((time.monotonic() - t0) * 1000)

    reply_language = detect_language(reply)

    t0 = time.monotonic()
    await save_message(db, conv.id, "user", transcript)
    await save_message(
        db, conv.id, "assistant", reply,
        used_search=bool(search_results),
        search_query=search_query_str,
    )
    timing["persist_ms"] = int((time.monotonic() - t0) * 1000)
    timing["total_ms"]   = int((time.monotonic() - t_start) * 1000)

    msg_count = await get_message_count(db, conv.id)

    return PipelineResult(
        transcript=transcript,
        transcript_language=detected_lang,
        transcript_language_code="hi" if detected_lang == "hindi" else "en",
        audio_duration=None,
        conversation_id=conv.id,
        message_count=msg_count,
        history_turns_used=len(history) // 2,
        used_web_search=bool(search_results),
        search_query=search_query_str,
        reply=reply,
        reply_language=reply_language,
        timing=timing,
    )
