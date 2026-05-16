"""
database/models.py — SQLAlchemy ORM models for JARVIS memory storage.

Tables:
  conversations  — one row per chat session
  messages       — individual turns (user + assistant) linked to a conversation
"""

from datetime import datetime
from sqlalchemy import (
    Column, String, Text, DateTime, ForeignKey, Integer, Boolean, Index
)
from sqlalchemy.orm import relationship
from database.db import Base


class Conversation(Base):
    """Represents a single conversation session."""
    __tablename__ = "conversations"

    id          = Column(String(36), primary_key=True)          # UUID
    title       = Column(String(255), nullable=True)            # auto-generated summary title
    language    = Column(String(20), default="hinglish")        # detected language
    created_at  = Column(DateTime, default=datetime.utcnow)
    updated_at  = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active   = Column(Boolean, default=True)

    # One conversation → many messages
    messages = relationship(
        "Message", back_populates="conversation",
        cascade="all, delete-orphan", order_by="Message.created_at"
    )

    def __repr__(self):
        return f"<Conversation id={self.id} title={self.title!r}>"


class Message(Base):
    """Represents a single message turn in a conversation."""
    __tablename__ = "messages"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=False)
    role            = Column(String(20), nullable=False)    # 'user' | 'assistant' | 'system'
    content         = Column(Text, nullable=False)
    language        = Column(String(20), default="hinglish")
    used_search     = Column(Boolean, default=False)        # did this turn trigger a web search?
    search_query    = Column(Text, nullable=True)           # the query that was searched
    created_at      = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")

    # Speed up fetching recent messages for a conversation
    __table_args__ = (
        Index("ix_messages_conversation_created", "conversation_id", "created_at"),
    )

    def __repr__(self):
        return f"<Message id={self.id} role={self.role} conv={self.conversation_id}>"
