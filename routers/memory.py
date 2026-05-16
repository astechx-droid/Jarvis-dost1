"""
routers/memory.py — Conversation memory management endpoints.

GET    /memory/conversations              — List all conversations
GET    /memory/conversations/{id}         — Get conversation + messages
DELETE /memory/conversations/{id}         — Delete a conversation
DELETE /memory/conversations/{id}/clear   — Clear messages only
GET    /memory/conversations/{id}/history — Get context-ready message history
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database.db import get_db
from database.models import Conversation, Message
from services.memory_service import (
    list_conversations,
    get_conversation,
    delete_conversation,
    clear_conversation_messages,
    get_conversation_history,
    get_message_count,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/memory", tags=["Memory"])


# ── Response Schemas ──────────────────────────────────────────────────────────

class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    language: str
    used_search: bool
    search_query: Optional[str]
    created_at: str

    model_config = {"from_attributes": True}


class ConversationOut(BaseModel):
    id: str
    title: Optional[str]
    language: str
    created_at: str
    updated_at: str
    message_count: Optional[int] = None

    model_config = {"from_attributes": True}


class ConversationDetail(ConversationOut):
    messages: List[MessageOut] = []


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/conversations", response_model=List[ConversationOut], summary="List all conversations")
async def list_convs(
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """Return a paginated list of conversations, newest first."""
    convs = await list_conversations(db, limit=limit, offset=offset)
    result = []
    for c in convs:
        count = await get_message_count(db, c.id)
        result.append(ConversationOut(
            id=c.id,
            title=c.title,
            language=c.language,
            created_at=c.created_at.isoformat(),
            updated_at=c.updated_at.isoformat(),
            message_count=count,
        ))
    return result


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail, summary="Get conversation with messages")
async def get_conv(conversation_id: str, db: AsyncSession = Depends(get_db)):
    """Return a conversation and all its messages."""
    conv = await get_conversation(db, conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )
    messages = list(result.scalars().all())

    return ConversationDetail(
        id=conv.id,
        title=conv.title,
        language=conv.language,
        created_at=conv.created_at.isoformat(),
        updated_at=conv.updated_at.isoformat(),
        message_count=len(messages),
        messages=[
            MessageOut(
                id=m.id,
                role=m.role,
                content=m.content,
                language=m.language,
                used_search=m.used_search,
                search_query=m.search_query,
                created_at=m.created_at.isoformat(),
            )
            for m in messages
        ],
    )


@router.get("/conversations/{conversation_id}/history", summary="Get context-ready message history")
async def get_history(
    conversation_id: str,
    max_turns: int = 20,
    db: AsyncSession = Depends(get_db),
):
    """
    Returns conversation history in the same format used for Groq context building.
    Useful for debugging and voice assistant context inspection.
    """
    conv = await get_conversation(db, conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    history = await get_conversation_history(db, conversation_id, max_turns=max_turns)
    return {"conversation_id": conversation_id, "turns": len(history), "history": history}


@router.delete("/conversations/{conversation_id}", summary="Delete a conversation")
async def delete_conv(conversation_id: str, db: AsyncSession = Depends(get_db)):
    """Soft-delete a conversation."""
    deleted = await delete_conversation(db, conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"message": f"Conversation {conversation_id} deleted.", "conversation_id": conversation_id}


@router.delete("/conversations/{conversation_id}/clear", summary="Clear conversation messages")
async def clear_conv(conversation_id: str, db: AsyncSession = Depends(get_db)):
    """Delete all messages in a conversation but keep the conversation itself."""
    conv = await get_conversation(db, conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    count = await clear_conversation_messages(db, conversation_id)
    return {
        "message": f"Cleared {count} messages from conversation {conversation_id}.",
        "conversation_id": conversation_id,
        "messages_deleted": count,
    }
