"""
services/groq_service.py — Groq API integration for JARVIS.

Handles:
  - Standard (non-streaming) chat completions
  - Streaming completions (for future WebSocket support)
  - Error handling and retries
"""

import ssl
import logging
from typing import List, Dict
import httpx
from groq import AsyncGroq, APIError, RateLimitError, APIConnectionError
from config import settings
from core.context import trim_history_for_context

logger = logging.getLogger(__name__)

# ── Groq client (singleton) ───────────────────────────────────────────────────
_client = None


def _make_http_client() -> httpx.AsyncClient:
    """
    Build an httpx AsyncClient using Python's built-in SSL context.
    This bypasses the broken certifi installation in this environment.
    """
    ssl_ctx = ssl.create_default_context()
    return httpx.AsyncClient(verify=ssl_ctx, timeout=60.0)


def get_groq_client() -> AsyncGroq:
    """Return (or create) the shared AsyncGroq client."""
    global _client
    if _client is None:
        _client = AsyncGroq(
            api_key=settings.groq_api_key,
            http_client=_make_http_client(),
        )
    return _client


# ── Standard completion ───────────────────────────────────────────────────────
async def chat_completion(messages: List[Dict]) -> str:
    """
    Send messages to Groq and return the assistant's reply as a string.

    Args:
        messages: Full message list (system prompt + history + current user msg).

    Returns:
        The assistant's text response.

    Raises:
        ValueError: On Groq API errors.
    """
    client = get_groq_client()

    # Safety trim before sending
    messages = trim_history_for_context(messages)

    try:
        response = await client.chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            temperature=0.75,       # balanced: creative but focused
            max_tokens=1024,        # generous limit for detailed answers
            top_p=0.9,
            stream=False,
        )
        reply = response.choices[0].message.content
        logger.info(
            "Groq response received | model=%s | tokens_used=%s",
            settings.groq_model,
            response.usage.total_tokens if response.usage else "unknown",
        )
        return reply.strip()

    except RateLimitError:
        logger.warning("Groq rate limit hit.")
        raise ValueError(
            "Rate limit reached. Please wait a moment, Mr Aryan, and try again."
        )
    except APIConnectionError:
        logger.error("Could not connect to Groq API.")
        raise ValueError(
            "Connection issue with Groq API. Please check your network."
        )
    except APIError as e:
        logger.error("Groq API error: %s", e)
        raise ValueError(f"Groq API error: {e.message}")


# ── Streaming completion (future WebSocket use) ───────────────────────────────
async def chat_completion_stream(messages: List[Dict]):
    """
    Async generator that yields text chunks from a streaming Groq response.
    Designed to be consumed by a WebSocket endpoint.

    Usage:
        async for chunk in chat_completion_stream(messages):
            await websocket.send_text(chunk)
    """
    client = get_groq_client()
    messages = trim_history_for_context(messages)

    try:
        stream = await client.chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            temperature=0.75,
            max_tokens=1024,
            top_p=0.9,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    except RateLimitError:
        yield "[Rate limit reached. Please try again shortly, Mr Aryan.]"
    except APIConnectionError:
        yield "[Connection error. Please check your network.]"
    except APIError as e:
        yield f"[Groq API error: {e.message}]"
