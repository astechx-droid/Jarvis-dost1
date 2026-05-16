"""
services/memory_service.py — Conversation memory management for JARVIS.

Provides functions to:
  - Create / retrieve conversation sessions
  - Save messages (user + assistant turns)
  - Load history for context building
  - List and delete conversations
"""

import uuid
import logging
from datetime import datetime
from typing import List, Optional
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import Conversation, Message
from core.personality import detect_language

logger = logging.getLogger(__name__)


# ── Conversation CRUD ─────────────────────────────────────────────────────────

async def create_conversation(db: AsyncSession, title: Optional[str] = None) -> Conversation:
    """Create a new conversation session and return it."""
    conv = Conversation(
        id=str(uuid.uuid4()),
        title=title or f"Conversation {datetime.utcnow().strftime('%d %b %Y %H:%M')}",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(conv)
    await db.flush()
    logger.info("New conversation created | id=%s", conv.id)
    return conv


async def get_conversation(db: AsyncSession, conversation_id: str) -> Optional[Conversation]:
    """Fetch a conversation by ID."""
    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    return result.scalar_one_or_none()


async def get_or_create_conversation(
    db: AsyncSession, conversation_id: Optional[str]
) -> Conversation:
    """
    Return an existing conversation if ID is provided and found,
    otherwise create a new one.
    """
    if conversation_id:
        conv = await get_conversation(db, conversation_id)
        if conv:
            return conv
    return await create_conversation(db)


async def list_conversations(
    db: AsyncSession, limit: int = 20, offset: int = 0
) -> List[Conversation]:
    """List recent conversations, newest first."""
    result = await db.execute(
        select(Conversation)
        .where(Conversation.is_active == True)
        .order_by(Conversation.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(result.scalars().all())


async def delete_conversation(db: AsyncSession, conversation_id: str) -> bool:
    """Soft-delete a conversation (marks is_active=False)."""
    result = await db.execute(
        update(Conversation)
        .where(Conversation.id == conversation_id)
        .values(is_active=False)
    )
    return result.rowcount > 0


async def clear_conversation_messages(db: AsyncSession, conversation_id: str) -> int:
    """Hard-delete all messages in a conversation. Returns number of rows deleted."""
    result = await db.execute(
        delete(Message).where(Message.conversation_id == conversation_id)
    )
    return result.rowcount


# ── Message CRUD ──────────────────────────────────────────────────────────────

async def save_message(
    db: AsyncSession,
    conversation_id: str,
    role: str,
    content: str,
    used_search: bool = False,
    search_query: Optional[str] = None,
) -> Message:
    """
    Save a single message to the database.

    Args:
        conversation_id: The parent conversation UUID.
        role:            'user' | 'assistant' | 'system'
        content:         The message text.
        used_search:     True if this turn triggered a web search.
        search_query:    The query used (if searched).
    """
    lang = detect_language(content) if role == "user" else "auto"
    msg = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        language=lang,
        used_search=used_search,
        search_query=search_query,
        created_at=datetime.utcnow(),
    )
    db.add(msg)

    # Bump conversation updated_at
    await db.execute(
        update(Conversation)
        .where(Conversation.id == conversation_id)
        .values(updated_at=datetime.utcnow())
    )
    await db.flush()
    return msg


async def get_conversation_history(
    db: AsyncSession,
    conversation_id: str,
    max_turns: int = 20,
) -> List[dict]:
    """
    Load the last `max_turns` user+assistant message pairs for a conversation.

    Returns:
        List of {"role": ..., "content": ...} dicts, oldest first,
        ready to be passed to the context builder.
    """
    result = await db.execute(
        select(Message)
        .where(
            Message.conversation_id == conversation_id,
            Message.role.in_(["user", "assistant"]),
        )
        .order_by(Message.created_at.desc())
        .limit(max_turns * 2)
    )
    messages = list(result.scalars().all())
    messages.reverse()

    return [{"role": m.role, "content": m.content} for m in messages]


async def get_message_count(db: AsyncSession, conversation_id: str) -> int:
    """Return total number of messages in a conversation."""
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
    )
    return len(result.scalars().all())
