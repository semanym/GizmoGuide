"""Structured RAG debug snapshots for trace and optional API output."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.knowledge.models import RetrievedChunk


def _round_score(score: float | None) -> float | None:
    if score is None:
        return None
    return round(float(score), 6)


def _preview(text: str, max_chars: int = 160) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[:max_chars]}..."


@dataclass
class RAGDebugChunk:
    """A compact, serialisable view of one retrieved chunk at a pipeline stage."""

    rank: int
    chunk_id: str
    title: str
    category: str
    score: float | None
    retrieval_path: str
    source: str
    tags: list[str] = field(default_factory=list)
    content_preview: str = ""

    @classmethod
    def from_chunk(cls, chunk: RetrievedChunk, rank: int) -> "RAGDebugChunk":
        return cls(
            rank=rank,
            chunk_id=chunk.chunk_id,
            title=chunk.title,
            category=chunk.category,
            score=_round_score(chunk.score),
            retrieval_path=chunk.retrieval_path,
            source=chunk.source,
            tags=chunk.tags,
            content_preview=_preview(chunk.content),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "title": self.title,
            "category": self.category,
            "score": self.score,
            "retrieval_path": self.retrieval_path,
            "source": self.source,
            "tags": self.tags,
            "content_preview": self.content_preview,
        }


@dataclass
class RRFDebugChunk:
    """RRF explanation for a fused candidate."""

    rank: int
    chunk_id: str
    title: str
    category: str
    rrf_score: float
    sources: list[str]
    source_ranks: dict[str, int]
    source_scores: dict[str, float | None]
    content_preview: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "title": self.title,
            "category": self.category,
            "rrf_score": _round_score(self.rrf_score),
            "sources": self.sources,
            "source_ranks": self.source_ranks,
            "source_scores": self.source_scores,
            "content_preview": self.content_preview,
        }


@dataclass
class RerankDebugChunk:
    """Cross-encoder rerank explanation for one candidate."""

    rank: int
    chunk_id: str
    title: str
    category: str
    original_index: int
    rrf_rank: int | None
    rrf_score: float | None
    relevance_score: float | None
    content_preview: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "title": self.title,
            "category": self.category,
            "original_index": self.original_index,
            "rrf_rank": self.rrf_rank,
            "rrf_score": _round_score(self.rrf_score),
            "relevance_score": _round_score(self.relevance_score),
            "content_preview": self.content_preview,
        }


@dataclass
class RAGDebugSnapshot:
    """End-to-end explainability snapshot for one RAG search."""

    original_query: str
    search_query: str
    category: str | None
    top_k: int
    recall_top_k: int
    rrf_k: int
    query_rewritten: bool = False
    detected_category: str | None = None
    tokenized_query: str = ""
    stages: list[str] = field(default_factory=list)
    bm25_results: list[RAGDebugChunk] = field(default_factory=list)
    vector_results: list[RAGDebugChunk] = field(default_factory=list)
    rrf_results: list[RRFDebugChunk] = field(default_factory=list)
    rerank_input: list[RAGDebugChunk] = field(default_factory=list)
    rerank_results: list[RerankDebugChunk] = field(default_factory=list)
    rerank_status: str = "skipped"
    rerank_reason: str | None = None
    final_results: list[RAGDebugChunk] = field(default_factory=list)

    def to_dict(self, max_items: int | None = None) -> dict[str, Any]:
        def limit(items: list[Any]) -> list[Any]:
            selected = items if max_items is None else items[:max_items]
            return [item.to_dict() for item in selected]

        return {
            "original_query": self.original_query,
            "search_query": self.search_query,
            "category": self.category,
            "detected_category": self.detected_category,
            "query_rewritten": self.query_rewritten,
            "tokenized_query": self.tokenized_query,
            "top_k": self.top_k,
            "recall_top_k": self.recall_top_k,
            "rrf_k": self.rrf_k,
            "stages": self.stages,
            "counts": {
                "bm25": len(self.bm25_results),
                "vector": len(self.vector_results),
                "rrf": len(self.rrf_results),
                "rerank_input": len(self.rerank_input),
                "rerank": len(self.rerank_results),
                "final": len(self.final_results),
            },
            "bm25_results": limit(self.bm25_results),
            "vector_results": limit(self.vector_results),
            "rrf_results": limit(self.rrf_results),
            "rerank": {
                "status": self.rerank_status,
                "reason": self.rerank_reason,
                "input": limit(self.rerank_input),
                "results": limit(self.rerank_results),
            },
            "final_results": limit(self.final_results),
        }

    def to_trace_summary(self, max_items: int = 5) -> dict[str, Any]:
        """Small enough for Langfuse span output while preserving key evidence."""
        data = self.to_dict(max_items=max_items)
        data["bm25_results"] = [
            {"rank": r["rank"], "chunk_id": r["chunk_id"], "title": r["title"], "score": r["score"]}
            for r in data["bm25_results"]
        ]
        data["vector_results"] = [
            {"rank": r["rank"], "chunk_id": r["chunk_id"], "title": r["title"], "score": r["score"]}
            for r in data["vector_results"]
        ]
        data["rrf_results"] = [
            {
                "rank": r["rank"],
                "chunk_id": r["chunk_id"],
                "title": r["title"],
                "rrf_score": r["rrf_score"],
                "source_ranks": r["source_ranks"],
            }
            for r in data["rrf_results"]
        ]
        data["rerank"]["input"] = [
            {"rank": r["rank"], "chunk_id": r["chunk_id"], "title": r["title"], "score": r["score"]}
            for r in data["rerank"]["input"]
        ]
        data["rerank"]["results"] = [
            {
                "rank": r["rank"],
                "chunk_id": r["chunk_id"],
                "title": r["title"],
                "original_index": r["original_index"],
                "relevance_score": r["relevance_score"],
            }
            for r in data["rerank"]["results"]
        ]
        data["final_results"] = [
            {"rank": r["rank"], "chunk_id": r["chunk_id"], "title": r["title"], "score": r["score"], "path": r["retrieval_path"]}
            for r in data["final_results"]
        ]
        return data


def debug_chunks(chunks: list[RetrievedChunk], limit: int | None = None) -> list[RAGDebugChunk]:
    selected = chunks if limit is None else chunks[:limit]
    return [RAGDebugChunk.from_chunk(chunk, rank=i) for i, chunk in enumerate(selected, start=1)]
