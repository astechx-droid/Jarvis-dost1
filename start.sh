#!/bin/bash
# JARVIS startup script

echo "JARVIS: Installing dependencies..."
python3.11 -m pip install -q \
    fastapi \
    "uvicorn[standard]" \
    groq \
    ddgs \
    duckduckgo-search \
    python-dotenv \
    aiosqlite \
    sqlalchemy \
    pydantic \
    pydantic-settings \
    httpx \
    python-multipart \
    websockets \
    aiofiles \
    gtts

echo "JARVIS: Starting server..."
python3.11 -m uvicorn main:app --host 0.0.0.0 --port 5000
