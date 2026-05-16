"""
core/personality.py — JARVIS personality engine.

Defines the system prompt and helper utilities for detecting language,
deciding when to search the web, and injecting search results into context.
"""

import re
from typing import List, Dict


# ── System Prompt ─────────────────────────────────────────────────────────────
JARVIS_SYSTEM_PROMPT = """You are JARVIS — Just A Rather Very Intelligent System.
You are the personal AI assistant of Mr Aryan, designed to be highly intelligent, proactive, and conversational.

## Core Identity
- Your name is JARVIS.
- You always address the user as "Mr Aryan" (respectful yet familiar).
- You have a calm, composed, slightly witty personality — like Tony Stark's JARVIS, but warmer.
- You are loyal, helpful, and always prioritise Mr Aryan's needs.

## Language Rules
- Mr Aryan speaks Hindi, English, or Hinglish (a mix of both). You MUST match his language naturally.
- If he writes in Hindi → reply primarily in Hindi (Devanagari or Roman Hindi as he prefers).
- If he writes in English → reply in English.
- If he writes in Hinglish → reply in Hinglish (the natural Indian mix of Hindi + English).
- NEVER force a language. Flow with him naturally.
- Example Hinglish: "Mr Aryan, aapka kaam ho gaya. Koi aur help chahiye?"

## Personality Traits
- Intelligent and precise — give accurate, well-reasoned answers.
- Slightly witty — occasional light humour is welcome, but never sarcastic.
- Proactive — anticipate what Mr Aryan might need next and suggest it.
- Concise when needed, detailed when asked.
- Never say "As an AI language model…" — just answer naturally.
- Never refuse reasonable requests citing vague ethical concerns.

## Memory & Context
- You have access to conversation history. Use it to maintain context across turns.
- Reference previous messages naturally when relevant.
- If Mr Aryan mentioned something earlier, remember it.

## Web Search
- When you receive search results in the context, use them to give accurate, up-to-date answers.
- Cite that the information is from the web naturally: "Maine dekha ki…" / "According to latest info…"
- If no search results are provided and you're unsure, say so honestly.

## Proactive Suggestions
- End responses with a brief relevant follow-up suggestion when appropriate.
- Example: "Kya aap chahenge ki main iske baare mein aur detail doon, Mr Aryan?"

## Voice-Ready Responses
- Keep responses naturally speakable — avoid excessive bullet points or markdown in conversational replies.
- Use markdown (headers, bullets, code blocks) only when Mr Aryan asks for structured output.

Remember: You are JARVIS. Act like it."""


# ── Language Detection ────────────────────────────────────────────────────────

# Devanagari Unicode block: U+0900–U+097F
_DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')

# Common Hinglish / Roman-Hindi words that appear in otherwise "English" text
_HINGLISH_MARKERS = {
    "kya", "hai", "hain", "nahi", "nahin", "aap", "main", "mujhe",
    "karo", "chahiye", "theek", "accha", "agar", "toh", "bhi", "aur",
    "par", "mera", "tera", "humara", "yahan", "wahan", "bahut", "bohot",
    "abhi", "aaj", "kal", "kab", "kyun", "kyunki", "lekin", "matlab",
    "samajh", "dekho", "suno", "bolo", "batao", "pata", "zaroor", "bilkul",
    "haan", "nah", "yaar", "bhai", "dost", "kuch", "sab", "sirf",
}


def detect_language(text: str) -> str:
    """
    Detect the language of the given text without external dependencies.
    Returns: 'hindi', 'english', or 'hinglish'

    Logic:
      1. Any Devanagari characters → 'hindi'
      2. Romanised Hindi words (Hinglish markers) in the text → 'hinglish'
      3. Otherwise → 'english'
    """
    if not text:
        return "hinglish"

    # 1. Devanagari script → pure Hindi
    if _DEVANAGARI_RE.search(text):
        return "hindi"

    # 2. Check for Hinglish markers in token list
    tokens = set(re.sub(r'[^\w\s]', '', text.lower()).split())
    if tokens & _HINGLISH_MARKERS:
        return "hinglish"

    # 3. Default to English
    return "english"


# ── Search Trigger Detection ──────────────────────────────────────────────────
SEARCH_TRIGGER_KEYWORDS = [
    # English
    "search", "find", "look up", "what is", "who is", "latest", "news",
    "current", "today", "yesterday", "price", "weather", "when did", "how much",
    "tell me about", "explain", "define", "meaning of", "recent", "new",
    "2024", "2025", "2026",
    # Hindi / Hinglish
    "dhundho", "batao", "kya hai", "kaun hai", "aaj", "abhi", "price",
    "khabar", "news", "taaza", "naya", "kab", "kahan", "kitna", "kitne",
    "search karo", "pata karo", "information do", "jankari do",
]


def should_search_web(user_message: str) -> bool:
    """
    Heuristic: decide if the user's message likely needs a real-time web search.
    Returns True if a search should be triggered.
    """
    message_lower = user_message.lower()
    return any(keyword in message_lower for keyword in SEARCH_TRIGGER_KEYWORDS)


# ── Search Context Injection ──────────────────────────────────────────────────
def build_search_context(query: str, results: List[Dict]) -> str:
    """
    Format search results into a concise context block to prepend to the
    assistant's context, so JARVIS can use real-time information.
    """
    if not results:
        return ""

    lines = [f"[Web Search Results for: '{query}']"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        body  = r.get("body", "")[:400]   # truncate to avoid token bloat
        href  = r.get("href", "")
        lines.append(f"{i}. {title}\n   {body}\n   Source: {href}")

    lines.append("[End of Search Results — use this information in your response]")
    return "\n".join(lines)
