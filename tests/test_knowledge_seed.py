"""Tests for offline cache-fed knowledge seeding (app.knowledge.seed)."""
from __future__ import annotations

import json
from pathlib import Path

from app.config.settings import Settings
from app.knowledge.embedding_cache import embed_text, text_sha256
from app.knowledge.seed import seed_knowledge


class FakeStore:
    """Minimal KnowledgeStore stand-in that records what got upserted."""

    def __init__(self, initial_count: int = 0):
        self._count = initial_count
        self.upserted = []

    def count(self) -> int:
        return self._count

    def upsert_chunks(self, chunks) -> int:
        self.upserted = list(chunks)
        self._count = len(self.upserted)
        return len(self.upserted)


class RaisingEmbedder:
    """Embedder that fails if called — proves the cache path avoids the API."""

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        raise AssertionError("embedder should not be called when cache is complete")


class RecordingEmbedder:
    """Embedder that returns a marker vector and counts how many texts it saw."""

    def __init__(self, dimensions: int = 4):
        self.dimensions = dimensions
        self.embedded_texts: list[str] = []

    def embed(self, texts):
        self.embedded_texts.extend(texts)
        return [[float(len(t) % 7) + 1.0] * self.dimensions for t in texts]


def _corpus() -> list[dict]:
    return [
        {"chunk_id": "c1", "category": "cat", "title": "标题一", "content": "内容一", "tags": ["a"], "source": "test"},
        {"chunk_id": "c2", "category": "cat", "title": "标题二", "content": "内容二", "tags": ["b"], "source": "test"},
    ]


def _write_corpus(corpus_dir: Path, items: list[dict]) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "corpus.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")


def _write_cache(cache_path: Path, items: list[dict], dimensions: int = 4) -> None:
    vectors = {}
    for item in items:
        sha = text_sha256(embed_text(item))
        vectors[item["chunk_id"]] = {"text_sha256": sha, "embedding": [0.5] * dimensions}
    payload = {"model": "text-embedding-v3", "dimensions": dimensions, "vectors": vectors}
    cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _settings(corpus_dir: Path, cache_path: Path, dimensions: int = 4) -> Settings:
    return Settings(
        deepseek_api_key=None,
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-chat",
        llm_timeout_seconds=1,
        embedding_dimensions=dimensions,
        knowledge_corpus_dir=str(corpus_dir),
        knowledge_embeddings_path=str(cache_path),
    )


def test_seed_uses_cache_without_calling_api(tmp_path):
    corpus_dir = tmp_path / "knowledge"
    cache_path = tmp_path / "embeddings.json"
    corpus = _corpus()
    _write_corpus(corpus_dir, corpus)
    _write_cache(cache_path, corpus)

    store = FakeStore()
    embedder = RaisingEmbedder()

    count = seed_knowledge(store, embedder, _settings(corpus_dir, cache_path))

    assert count == 2
    assert embedder.calls == 0
    assert {c.chunk_id for c in store.upserted} == {"c1", "c2"}
    assert all(c.embedding == [0.5] * 4 for c in store.upserted)


def test_seed_embeds_only_missing_or_stale_chunks(tmp_path):
    corpus_dir = tmp_path / "knowledge"
    cache_path = tmp_path / "embeddings.json"
    corpus = _corpus()
    _write_corpus(corpus_dir, corpus)
    # Cache only c1 with a matching hash; c2 is absent -> only c2 should hit the API.
    _write_cache(cache_path, corpus[:1])

    store = FakeStore()
    embedder = RecordingEmbedder(dimensions=4)

    count = seed_knowledge(store, embedder, _settings(corpus_dir, cache_path))

    assert count == 2
    # Exactly one chunk (c2) was embedded live.
    assert embedder.embedded_texts == [embed_text(corpus[1])]


def test_seed_reembeds_when_text_changed(tmp_path):
    corpus_dir = tmp_path / "knowledge"
    cache_path = tmp_path / "embeddings.json"
    corpus = _corpus()
    _write_corpus(corpus_dir, corpus)
    _write_cache(cache_path, corpus)

    # Mutate c1's content on disk so its hash no longer matches the cache.
    corpus[0]["content"] = "内容一被改了"
    _write_corpus(corpus_dir, corpus)

    store = FakeStore()
    embedder = RecordingEmbedder(dimensions=4)

    seed_knowledge(store, embedder, _settings(corpus_dir, cache_path))

    assert embedder.embedded_texts == [embed_text(corpus[0])]


def test_seed_skips_when_store_not_empty(tmp_path):
    corpus_dir = tmp_path / "knowledge"
    cache_path = tmp_path / "embeddings.json"
    corpus = _corpus()
    _write_corpus(corpus_dir, corpus)
    _write_cache(cache_path, corpus)

    store = FakeStore(initial_count=5)
    embedder = RaisingEmbedder()

    count = seed_knowledge(store, embedder, _settings(corpus_dir, cache_path))

    assert count == 0
    assert embedder.calls == 0
    assert store.upserted == []
