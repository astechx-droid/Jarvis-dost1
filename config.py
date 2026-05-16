"""
config.py — Centralised configuration for JARVIS backend.
Reads all settings from environment variables / .env file.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # ── Groq ──────────────────────────────────────────────
    groq_api_key: str = Field(default="", env="GROQ_API_KEY")
    groq_model: str = Field("llama-3.3-70b-versatile", env="GROQ_MODEL")

    # ── Database — use JARVIS_DB_URL to avoid conflict with Replit's DATABASE_URL
    # Falls back to local SQLite if not set
    jarvis_db_url: str = Field(
        "sqlite+aiosqlite:///./jarvis_memory.db", env="JARVIS_DB_URL"
    )

    # ── Conversation ──────────────────────────────────────
    max_history_turns: int = Field(20, env="MAX_HISTORY_TURNS")

    # ── Web Search ────────────────────────────────────────
    max_search_results: int = Field(5, env="MAX_SEARCH_RESULTS")

    # ── App ───────────────────────────────────────────────
    app_host: str = Field("0.0.0.0", env="APP_HOST")
    app_port: int = Field(8000, env="APP_PORT")
    debug: bool = Field(False, env="DEBUG")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @property
    def database_url(self) -> str:
        return self.jarvis_db_url


# Singleton — import `settings` everywhere else
settings = Settings()
