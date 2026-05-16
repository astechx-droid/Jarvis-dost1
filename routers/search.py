"""
routers/search.py — Direct web search endpoints.

POST /search/web   — Trigger a web search and return raw results
POST /search/news  — Search DuckDuckGo News
"""

import logging
from typing import List

from fastapi import APIRouter
from pydantic import BaseModel, Field

from services.search_service import web_search, news_search, extract_search_query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["Search"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    max_results: int = Field(5, ge=1, le=20)

    model_config = {"json_schema_extra": {"example": {
        "query": "India vs Australia cricket 2025",
        "max_results": 5,
    }}}


class SearchResult(BaseModel):
    title: str
    href: str
    body: str


class SearchResponse(BaseModel):
    query: str
    cleaned_query: str
    results: List[SearchResult]
    result_count: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/web", response_model=SearchResponse, summary="Web search via DuckDuckGo")
async def search_web(request: SearchRequest):
    """
    Perform a real-time DuckDuckGo web search.
    Returns structured results with title, URL, and snippet.
    """
    cleaned = extract_search_query(request.query)
    results = await web_search(cleaned, max_results=request.max_results)

    return SearchResponse(
        query=request.query,
        cleaned_query=cleaned,
        results=[
            SearchResult(
                title=r.get("title", ""),
                href=r.get("href", ""),
                body=r.get("body", ""),
            )
            for r in results
        ],
        result_count=len(results),
    )


@router.post("/news", response_model=SearchResponse, summary="News search via DuckDuckGo")
async def search_news(request: SearchRequest):
    """
    Search DuckDuckGo News for recent articles on a topic.
    """
    cleaned = extract_search_query(request.query)
    results = await news_search(cleaned, max_results=request.max_results)

    normalised = []
    for r in results:
        normalised.append(SearchResult(
            title=r.get("title", ""),
            href=r.get("url", r.get("href", "")),
            body=r.get("body", r.get("excerpt", "")),
        ))

    return SearchResponse(
        query=request.query,
        cleaned_query=cleaned,
        results=normalised,
        result_count=len(normalised),
    )
