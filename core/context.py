"""
core/context.py — Context builder for JARVIS.

Assembles the list of messages sent to Groq, combining:
  1. The JARVIS system prompt
  2. Conversation history from the DB (last N turns)
  3. Optional web search results (injected as a system message)
  4. The current user message
"""

from typing import List, Dict, Optional
from core.personality import JARVIS_SYSTEM_PROMPT, build_search_context
from config import settings


def build_messages(
    history: List[Dict],
    user_message: str,
    search_results: Optional[List[Dict]] = None,
    search_query: Optional[str] = None,
) -> List[Dict]:
    """
    Build the full messages array for the Groq chat completion API.

    Args:
        history:        List of {"role": ..., "content": ...} dicts (recent turns).
        user_message:   The current user input.
        search_results: Optional list of DuckDuckGo result dicts.
        search_query:   The query used for the search (for display in context).

    Returns:
        A list of message dicts ready to be sent to Groq.
    """
    messages: List[Dict] = []

    # 1. Always start with the JARVIS system prompt
    messages.append({"role": "system", "content": JARVIS_SYSTEM_PROMPT})

    # 2. Inject web search results as a system message (if any)
    if search_results and search_query:
        search_context = build_search_context(search_query, search_results)
        if search_context:
            messages.append({"role": "system", "content": search_context})

    # 3. Append recent conversation history (limited to MAX_HISTORY_TURNS)
    max_turns = settings.max_history_turns
    # Each "turn" = 1 user + 1 assistant message → 2 items
    history_window = history[-(max_turns * 2):]
    messages.extend(history_window)

    # 4. Append the current user message
    messages.append({"role": "user", "content": user_message})

    return messages


def trim_history_for_context(
    messages: List[Dict],
    max_tokens_estimate: int = 6000,
    avg_chars_per_token: int = 4,
) -> List[Dict]:
    """
    Safety trim: if estimated token count is too high, drop oldest history
    messages (preserving system prompt + current user message).

    This is a lightweight heuristic — Groq will also enforce its own limits.
    """
    max_chars = max_tokens_estimate * avg_chars_per_token

    total_chars = sum(len(m["content"]) for m in messages)
    if total_chars <= max_chars:
        return messages

    # Separate: system messages at the front, last user message at the back
    system_msgs  = [m for m in messages if m["role"] == "system"]
    history_msgs = [m for m in messages if m["role"] != "system"]

    # Drop from the oldest history until we're under the limit
    while history_msgs and total_chars > max_chars:
        removed = history_msgs.pop(0)
        total_chars -= len(removed["content"])

    return system_msgs + history_msgs
