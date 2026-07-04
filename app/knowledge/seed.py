"""Seed the knowledge base from local JSON corpus + a persisted embedding cache.

Startup seeding is meant to be offline and cheap: vectors come from the
committed cache file (built by ``scripts/knowledge/build_embeddings.py``). Only
chunks that are missing from the cache or whose text changed fall back to a live
embedding call, so a fully-built cache means zero API calls at startup.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import jieba

from app.config.settings import Settings, get_settings
from app.embedding.client import EmbeddingClient
from app.knowledge.embedding_cache import EmbeddingCache, embed_text, resolve_path, text_sha256
from app.knowledge.models import KnowledgeChunk
from app.knowledge.store import KnowledgeStore

logger = logging.getLogger(__name__)


def tokenize_chinese(text: str) -> str:
    """Tokenize Chinese text using jieba."""
    tokens = jieba.lcut(text)
    return " ".join(t.strip() for t in tokens if t.strip())


def load_seed_knowledge(corpus_dir: Path) -> list[dict]:
    """Load seed knowledge from all JSON files in the corpus directory."""
    if not corpus_dir.exists():
        logger.warning("Knowledge corpus directory not found at %s", corpus_dir)
        return []

    documents: list[dict] = []
    for file_path in sorted(corpus_dir.glob("*.json")):
        with open(file_path, "r", encoding="utf-8") as f:
            file_documents = json.load(f)
        if not isinstance(file_documents, list):
            logger.warning("Skipping non-list knowledge file: %s", file_path)
            continue
        documents.extend(file_documents)
        logger.info("Loaded %d knowledge chunks from %s", len(file_documents), file_path.name)

    return documents


def _embed_with_cache(
    seed_data: list[dict],
    embedder: EmbeddingClient,
    cache: EmbeddingCache,
) -> dict[str, list[float]]:
    """Resolve an embedding for every chunk, using the cache and calling the API
    only for chunks that are missing or whose text changed."""
    embeddings: dict[str, list[float]] = {}
    misses: list[dict] = []

    for item in seed_data:
        chunk_id = item["chunk_id"]
        sha = text_sha256(embed_text(item))
        cached = cache.get(chunk_id, sha)
        if cached is not None:
            embeddings[chunk_id] = cached
        else:
            misses.append(item)

    if misses:
        logger.info("Embedding cache miss for %d chunks; calling embedding API", len(misses))
        batch_size = 5
        for i in range(0, len(misses), batch_size):
            batch = misses[i : i + batch_size]
            texts = [embed_text(item) for item in batch]
            try:
                vectors = embedder.embed(texts)
            except Exception as exc:
                logger.error("Embedding generation failed for batch %d: %s", i // batch_size, exc)
                continue
            for item, vector in zip(batch, vectors):
                sha = text_sha256(embed_text(item))
                cache.put(item["chunk_id"], sha, vector)
                embeddings[item["chunk_id"]] = vector

    return embeddings


def seed_knowledge(
    store: KnowledgeStore,
    embedder: EmbeddingClient,
    settings: Settings | None = None,
) -> int:
    """Seed the knowledge base with local corpus data if it's empty.

    Returns the number of chunks seeded.
    """
    settings = settings or get_settings()

    existing_count = store.count()
    if existing_count > 0:
        logger.info("Knowledge base already has %d chunks, skipping seed", existing_count)
        return 0

    corpus_dir = resolve_path(settings.knowledge_corpus_dir)
    seed_data = load_seed_knowledge(corpus_dir)
    if not seed_data:
        logger.warning("No knowledge data to seed")
        return 0

    logger.info("Seeding %d knowledge chunks...", len(seed_data))

    cache = EmbeddingCache.load(
        resolve_path(settings.knowledge_embeddings_path),
        model=settings.dashscope_embedding_model,
        dimensions=settings.embedding_dimensions,
    )
    embeddings = _embed_with_cache(seed_data, embedder, cache)

    chunks: list[KnowledgeChunk] = []
    for item in seed_data:
        embedding = embeddings.get(item["chunk_id"])
        if embedding is None:
            logger.warning("No embedding available for chunk %s; skipping", item["chunk_id"])
            continue
        content_text = f"{item['title']} {item['content']}"
        chunks.append(
            KnowledgeChunk(
                chunk_id=item["chunk_id"],
                category=item["category"],
                title=item["title"],
                content=item["content"],
                tokenized=tokenize_chinese(content_text),
                tags=item.get("tags", []),
                source=item.get("source", "mock"),
                embedding=embedding,
            )
        )

    count = store.upsert_chunks(chunks)
    logger.info("Seeded %d knowledge chunks into the knowledge base", count)
    return count
