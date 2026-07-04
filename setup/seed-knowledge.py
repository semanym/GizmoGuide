"""One-shot seed script: run inside the app container to populate the knowledge base."""
from __future__ import annotations

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from app.config.settings import get_settings
from app.embedding.client import EmbeddingClient
from app.knowledge.seed import seed_knowledge
from app.knowledge.store import KnowledgeStore


def main() -> None:
    settings = get_settings()

    if not settings.dashscope_api_key:
        print("ERROR: DASHSCOPE_API_KEY is not set. Cannot generate embeddings.")
        sys.exit(1)

    database_url = settings.rag_database_url
    if not database_url:
        print("ERROR: RAG_DATABASE_URL is not set.")
        sys.exit(1)

    print(f"==> Ensuring schema on {database_url}...")
    store = KnowledgeStore(database_url)
    store.ensure_schema()

    print("==> Generating embeddings and seeding knowledge...")
    embedder = EmbeddingClient(
        settings.dashscope_api_key,
        model=settings.dashscope_embedding_model,
        dimensions=settings.embedding_dimensions,
    )
    count = seed_knowledge(store, embedder, settings)

    print(f"==> Done. {count} chunks seeded. Total: {store.count()}")


if __name__ == "__main__":
    main()
