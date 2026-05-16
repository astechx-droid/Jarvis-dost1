"""
services/search_service.py — Real-time web search via DuckDuckGo.

Uses duckduckgo-search (DDGS) to fetch live web results with no API key needed.
Results are returned as a list of dicts with keys: title, href, body.
"""

import logging
import asyncio
from typing import List, Optional

# Support both old and new package name
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

from config import settings

logger = logging.getLogger(__name__)


async def web_search(query: str, max_results: Optional[int] = None) -> List[dict]:
    """
    Perform a real-time DuckDuckGo web search.

    Args:
        query:       The search query string.
        max_results: Number of results to return (defaults to settings value).

    Returns:
        List of result dicts: [{"title": ..., "href": ..., "body": ...}, ...]
        Returns empty list if search fails.
    """
    if not query or not query.strip():
        return []

    n = max_results or settings.max_search_results

    try:
        results = await asyncio.to_thread(_sync_search, query, n)
        logger.info("DuckDuckGo search | query=%r | results=%d", query, len(results))
        return results

    except Exception as e:
        logger.warning("DuckDuckGo search failed | query=%r | error=%s", query, e)
        return []


def _sync_search(query: str, max_results: int) -> List[dict]:
    """Synchronous DuckDuckGo search (runs in a thread pool)."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return results


async def news_search(query: str, max_results: Optional[int] = None) -> List[dict]:
    """
    Search DuckDuckGo News for a topic.
    Useful for recent news, current events, and trending topics.

    Returns:
        List of news result dicts.
    """
    if not query or not query.strip():
        return []

    n = max_results or settings.max_search_results

    try:
        results = await asyncio.to_thread(_sync_news_search, query, n)
        logger.info("DuckDuckGo news search | query=%r | results=%d", query, len(results))
        return results

    except Exception as e:
        logger.warning("DuckDuckGo news search failed | query=%r | error=%s", query, e)
        return []


def _sync_news_search(query: str, max_results: int) -> List[dict]:
    """Synchronous DuckDuckGo news search (runs in a thread pool)."""
    with DDGS() as ddgs:
        results = list(ddgs.news(query, max_results=max_results))
    return results


def extract_search_query(user_message: str) -> str:
    """
    Clean up the user's message to form a good search query.
    Removes filler words and phrases to get a clean, searchable string.
    """
    fillers = [
        "search for", "find", "look up", "tell me about", "what is", "who is",
        "dhundho", "batao", "pata karo", "search karo", "kya hai", "kaun hai",
        "please", "can you", "kya aap", "mujhe batao", "information do",
    ]
    query = user_message.strip()
    for filler in fillers:
        query = query.lower().replace(filler, "").strip()

    return query.strip().capitalize() if query else user_message.strip()
