from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.chat import router as chat_router
from app.api.products import router as products_router
from app.api.recommend import router as recommend_router
from app.connectors.redis_product_metadata import get_product_metadata_store
from app.config.settings import get_settings

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT_DIR / "frontend"

app = FastAPI(title="GizmoGuide", version="0.1.0")
app.include_router(chat_router)
app.include_router(products_router)
app.include_router(recommend_router)

if FRONTEND_DIR.exists():
    app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="ui")


@app.on_event("startup")
def _seed_product_metadata() -> None:
    """Redis is the runtime source of product metadata; fail fast if unavailable."""
    store = get_product_metadata_store()
    total = store.ensure_seeded()
    logger.info("Product metadata ensured in Redis: %s records", total)


@app.on_event("startup")
def _ensure_rag_schema() -> None:
    """Create knowledge_base table if it doesn't exist (idempotent)."""
    try:
        settings = get_settings()
        if settings.rag_enabled:
            from app.knowledge.store import KnowledgeStore
            store = KnowledgeStore(settings.rag_database_url)
            store.ensure_schema()
            logger.info("RAG schema ensured")
    except Exception as exc:
        logger.warning("RAG schema init skipped: %s", exc)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "GizmoGuide"}


@app.get("/", include_in_schema=False)
def index() -> RedirectResponse:
    return RedirectResponse(url="/ui/")
