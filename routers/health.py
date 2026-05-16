"""
routers/health.py — Health check and system info endpoints.

GET /health         — Basic liveness check
GET /health/status  — Detailed system status (DB, Groq connectivity)
"""

import logging
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from database.db import get_db
from config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/", summary="Liveness check")
async def health():
    """Simple liveness probe — returns 200 if the server is running."""
    return {"status": "online", "service": "JARVIS AI Backend", "version": "1.0.0"}


@router.get("/status", summary="Detailed system status")
async def status(db: AsyncSession = Depends(get_db)):
    """
    Check connectivity to all subsystems:
    - SQLite database
    - Groq API (lightweight model list check)
    """
    report = {
        "service": "JARVIS AI Backend",
        "version": "1.0.0",
        "model": settings.groq_model,
        "database": "unknown",
        "groq_api": "unknown",
    }

    # ── Database check ─────────────────────────────────────────────────────
    try:
        await db.execute(text("SELECT 1"))
        report["database"] = "connected"
    except Exception as e:
        report["database"] = f"error: {e}"

    # ── Groq API check ─────────────────────────────────────────────────────
    try:
        from services.groq_service import get_groq_client
        client = get_groq_client()
        models_response = await client.models.list()
        model_ids = [m.id for m in models_response.data]
        if settings.groq_model in model_ids:
            report["groq_api"] = f"connected — model '{settings.groq_model}' available"
        else:
            report["groq_api"] = (
                f"connected — WARNING: model '{settings.groq_model}' not found. "
                f"Available: {', '.join(model_ids[:5])}"
            )
    except Exception as e:
        report["groq_api"] = f"error: {e}"

    overall = "healthy" if (
        "connected" in str(report.get("database", "")) and
        "connected" in str(report.get("groq_api", ""))
    ) else "degraded"

    report["overall"] = overall
    return report
