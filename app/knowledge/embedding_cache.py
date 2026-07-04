"""Persistent DashScope embedding cache for the knowledge corpus.

Vectors are stored separately from the corpus JSON (approach A): the corpus
stays human-readable, and dense vectors live in one committed cache file that
can be rebuilt incrementally offline.

Invalidation is per-chunk by (chunk_id + sha256 of the embed text). Editing a
chunk's title/content changes its hash, so the offline builder re-embeds only
the changed or new chunks and leaves everything else untouched.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]


def resolve_path(path_str: str) -> Path:
    """Resolve a configured path relative to the project root (or keep absolute)."""
    path = Path(path_str)
    return path if path.is_absolute() else (ROOT_DIR / path).resolve()


def embed_text(item: dict) -> str:
    """The exact text used to generate a chunk's embedding.

    Must stay identical between the offline builder and the seed path so that
    cached vectors line up with their chunks.
    """
    return f"{item['title']}\n{item['content']}"


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """In-memory view of the on-disk embedding cache file."""

    def __init__(self, model: str, dimensions: int, vectors: dict[str, dict[str, Any]] | None = None):
        self.model = model
        self.dimensions = dimensions
        self.vectors: dict[str, dict[str, Any]] = vectors or {}

    @classmethod
    def load(cls, path: Path, model: str, dimensions: int) -> "EmbeddingCache":
        if not path.exists():
            return cls(model=model, dimensions=dimensions)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            model=data.get("model", model),
            dimensions=int(data.get("dimensions", dimensions)),
            vectors=data.get("vectors", {}),
        )

    def get(self, chunk_id: str, sha: str) -> list[float] | None:
        """Return the cached embedding only if it matches the current text hash."""
        entry = self.vectors.get(chunk_id)
        if entry and entry.get("text_sha256") == sha:
            embedding = entry.get("embedding")
            if embedding:
                return embedding
        return None

    def put(self, chunk_id: str, sha: str, embedding: list[float]) -> None:
        self.vectors[chunk_id] = {"text_sha256": sha, "embedding": embedding}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model": self.model, "dimensions": self.dimensions, "vectors": self.vectors}
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
