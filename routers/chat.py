"""
routers/chat.py — Core chat endpoints for JARVIS.

POST /chat/         — Send a message and get JARVIS's reply
WS   /chat/stream   — WebSocket streaming endpoint (Phase 2 ready)
"""

import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from database.db import get_db
from services.memory_service import (
    get_or_create_conversation,
    get_conversation_history,
    save_message,
)
from services.groq_service import chat_completion, chat_completion_stream
from services.search_service import web_search, extract_search_query
from core.personality import detect_language, should_search_web
from core.context import build_messages
from config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


# ── Request / Response Schemas ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="User's message to JARVIS")
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID (omit to start new)")
    force_search: bool = Field(False, description="Force a web search even if heuristic says no")

    model_config = {"json_schema_extra": {"example": {
        "message": "Kya hai aaj ka weather Delhi mein?",
        "conversation_id": None,
        "force_search": False,
    }}}


class ChatResponse(BaseModel):
    reply: str
    conversation_id: str
    language_detected: str
    used_web_search: bool
    search_query: Optional[str]
    message_count: int


# ── POST /chat/ ────────────────────────────────────────────────────────────────

@router.post("/", response_model=ChatResponse, summary="Send a message to JARVIS")
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """
    Main chat endpoint.

    - Detects language (Hindi / English / Hinglish)
    - Decides if a web search is needed and performs it
    - Loads conversation history from SQLite
    - Builds context and sends to Groq
    - Saves both user and assistant messages to memory
    - Returns JARVIS's reply
    """
    user_message = request.message.strip()

    # 1. Detect language
    lang = detect_language(user_message)

    # 2. Get or create conversation
    conv = await get_or_create_conversation(db, request.conversation_id)

    # 3. Determine if web search is needed
    do_search = request.force_search or should_search_web(user_message)
    search_results = []
    search_query: Optional[str] = None

    if do_search:
        search_query = extract_search_query(user_message)
        logger.info("Web search triggered | query=%r", search_query)
        search_results = await web_search(search_query)

    # 4. Load conversation history
    history = await get_conversation_history(db, conv.id, max_turns=settings.max_history_turns)

    # 5. Build messages array for Groq
    messages = build_messages(
        history=history,
        user_message=user_message,
        search_results=search_results if search_results else None,
        search_query=search_query,
    )

    # 6. Call Groq
    try:
        reply = await chat_completion(messages)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e))

    # 7. Persist both turns to memory
    await save_message(db, conv.id, "user", user_message)
    await save_message(
        db, conv.id, "assistant", reply,
        used_search=do_search,
        search_query=search_query,
    )

    # 8. Message count (rough: history + new pair)
    msg_count = len(history) + 2

    return ChatResponse(
        reply=reply,
        conversation_id=conv.id,
        language_detected=lang,
        used_web_search=bool(search_results),
        search_query=search_query,
        message_count=msg_count,
    )


# ── WS /chat/stream/{conversation_id} — WebSocket (Phase 2 ready) ─────────────

@router.websocket("/stream/{conversation_id}")
async def chat_stream(
    websocket: WebSocket,
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    WebSocket streaming endpoint.

    Client sends: {"message": "..."}
    Server streams back text chunks, then sends {"done": true} when finished.
    """
    await websocket.accept()
    logger.info("WebSocket connection opened | conv=%s", conversation_id)

    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                user_message = payload.get("message", "").strip()
            except (json.JSONDecodeError, AttributeError):
                await websocket.send_json({"error": "Invalid JSON. Send {\"message\": \"...\"}"})
                continue

            if not user_message:
                await websocket.send_json({"error": "Empty message."})
                continue

            conv = await get_or_create_conversation(db, conversation_id)

            search_results = []
            search_query = None
            if should_search_web(user_message):
                search_query = extract_search_query(user_message)
                search_results = await web_search(search_query)

            history = await get_conversation_history(db, conv.id)
            messages = build_messages(
                history=history,
                user_message=user_message,
                search_results=search_results or None,
                search_query=search_query,
            )

            await save_message(db, conv.id, "user", user_message)
            await db.commit()

            full_reply = []
            async for chunk in chat_completion_stream(messages):
                await websocket.send_text(chunk)
                full_reply.append(chunk)

            complete_reply = "".join(full_reply)
            await save_message(
                db, conv.id, "assistant", complete_reply,
                used_search=bool(search_results), search_query=search_query,
            )
            await db.commit()

            await websocket.send_json({"done": True, "conversation_id": conv.id})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected | conv=%s", conversation_id)
    except Exception as e:
        logger.error("WebSocket error | conv=%s | error=%s", conversation_id, e)
        await websocket.send_json({"error": str(e)})
        await websocket.close()
