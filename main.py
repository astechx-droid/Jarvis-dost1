"""
main.py — JARVIS AI Backend Entry Point

Phase 1: REST API with conversation memory, Groq LLM, and DuckDuckGo search.
Future: WebSocket streaming (already wired in routers/chat.py).

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

from database.db import init_db
from routers import chat, memory, search, health, voice, tts, conversation

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the database on startup."""
    logger.info("JARVIS booting up… initialising database.")
    await init_db()
    logger.info("Database ready. JARVIS is online. Good day, Mr Aryan.")
    yield
    logger.info("JARVIS shutting down. Goodbye, Mr Aryan.")


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="JARVIS AI Backend",
    description=(
        "Phase 1 backend for JARVIS — a cloud-based AI assistant for Mr Aryan. "
        "Supports Hindi, English, and Hinglish conversations with persistent memory, "
        "real-time web search via DuckDuckGo, and Groq-powered LLM responses. "
        "WebSocket streaming is architecture-ready for Phase 2."
    ),
    version="1.0.0",
    contact={"name": "Mr Aryan's JARVIS", "url": "https://github.com"},
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── CORS (allow all origins for Phase 1 — restrict in production) ──────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(chat.router)
app.include_router(memory.router)
app.include_router(search.router)
app.include_router(voice.router)
app.include_router(tts.router)
app.include_router(conversation.router)

# ── Root — serve the chat UI ──────────────────────────────────────────────────
@app.get("/", tags=["Root"], response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("Unhandled exception: %s | path=%s", exc, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. JARVIS is investigating."},
    )


# ── Dev runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    from config import settings

    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
        log_level="info",
    )
