"""End-to-end RAG pipeline: query → dual recall → RRF → rerank → results."""
from __future__ import annotations

import logging
from typing import Any

from app.embedding.client import EmbeddingClient
from app.knowledge.debug import RAGDebugSnapshot, RerankDebugChunk, debug_chunks
from app.knowledge.models import RetrievedChunk
from app.knowledge.retrieval.dual_recall import dual_recall, tokenize_chinese
from app.knowledge.retrieval.rrf import reciprocal_rank_fusion_with_debug
from app.knowledge.store import KnowledgeStore
from app.reranker.client import RerankerClient
from app.tracing import trace_span

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Enterprise-grade RAG pipeline with query rewriting, dual recall, RRF fusion, and reranking."""

    def __init__(
        self,
        store: KnowledgeStore,
        embedder: EmbeddingClient,
        reranker: RerankerClient | None = None,
        query_rewriter: Any | None = None,
        recall_top_k: int = 10,
        rerank_top_n: int = 5,
        rrf_k: int = 60,
    ):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.query_rewriter = query_rewriter
        self.recall_top_k = recall_top_k
        self.rerank_top_n = rerank_top_n
        self.rrf_k = rrf_k

    def search(self, query: str, category: str | None = None, top_k: int | None = None) -> list[RetrievedChunk]:
        """Full RAG pipeline: query rewrite → dual recall → RRF → rerank.

        Returns the top-k most relevant knowledge chunks.
        """
        effective_top_k = top_k or self.rerank_top_n
        snapshot = RAGDebugSnapshot(
            original_query=query,
            search_query=query,
            category=category,
            top_k=effective_top_k,
            recall_top_k=self.recall_top_k,
            rrf_k=self.rrf_k,
        )

        with trace_span(
            "rag_pipeline",
            input_data={"query": query, "category": category, "top_k": effective_top_k},
        ) as (span, end_span):
            # Step 0: Query rewriting (optional)
            original_query = query
            if self.query_rewriter is not None:
                rewritten_query, detected_category = self.query_rewriter.rewrite(query)
                query = rewritten_query
                snapshot.detected_category = detected_category
                if category is None and detected_category:
                    category = detected_category
                logger.info(
                    "Query rewritten: '%s' → '%s' (category=%s)",
                    original_query, rewritten_query, category,
                )
            snapshot.search_query = query
            snapshot.category = category
            snapshot.query_rewritten = original_query != query
            snapshot.tokenized_query = tokenize_chinese(query)

            # Step 1: Dual recall
            snapshot.stages.append("dual_recall")
            bm25_results, vector_results = dual_recall(
                self.store, self.embedder, query,
                top_k=self.recall_top_k, category=category,
            )
            snapshot.bm25_results = debug_chunks(bm25_results)
            snapshot.vector_results = debug_chunks(vector_results)

            if not bm25_results and not vector_results:
                snapshot.stages.append("dual_recall_empty")
                end_span(output=snapshot.to_trace_summary())
                return []

            # Step 2: RRF fusion
            snapshot.stages.append("rrf")
            fused, rrf_debug = reciprocal_rank_fusion_with_debug(
                [("bm25", bm25_results), ("vector", vector_results)],
                k=self.rrf_k, top_n=self.recall_top_k,
            )
            snapshot.rrf_results = rrf_debug

            if not fused:
                snapshot.stages.append("rrf_empty")
                end_span(output=snapshot.to_trace_summary())
                return []

            # Step 3: Reranking (if reranker is available)
            if self.reranker and len(fused) > 1:
                snapshot.stages.append("rerank")
                snapshot.rerank_input = debug_chunks(fused)
                reranked, rerank_debug, rerank_status, rerank_reason = self._rerank_with_debug(query, fused)
                snapshot.rerank_results = rerank_debug
                snapshot.rerank_status = rerank_status
                snapshot.rerank_reason = rerank_reason
                final = reranked[:effective_top_k]
                stage = "dual_recall → rrf → rerank"
            else:
                snapshot.rerank_status = "skipped"
                if not self.reranker:
                    snapshot.rerank_reason = "reranker_not_configured"
                elif len(fused) <= 1:
                    snapshot.rerank_reason = "single_candidate"
                final = fused[:effective_top_k]
                stage = "dual_recall → rrf"

            snapshot.final_results = debug_chunks(final)

            end_span(
                output={"result_count": len(final), "stage_summary": stage, **snapshot.to_trace_summary()},
            )

            return final

    def _rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Apply cross-encoder reranking to the fused candidate list."""
        reranked, _debug, _status, _reason = self._rerank_with_debug(query, candidates)
        return reranked

    def _rerank_with_debug(
        self,
        query: str,
        candidates: list[RetrievedChunk],
    ) -> tuple[list[RetrievedChunk], list[RerankDebugChunk], str, str | None]:
        """Apply cross-encoder reranking and return candidate-level explanation."""
        with trace_span(
            "reranker",
            input_data={
                "query": query,
                "candidate_count": len(candidates),
                "candidates": [
                    {"index": i, "chunk_id": c.chunk_id, "title": c.title, "rrf_score": round(c.score, 6)}
                    for i, c in enumerate(candidates)
                ],
            },
        ) as (span, end_span):
            documents = [f"{c.title}\n{c.content}" for c in candidates]
            rrf_scores = {c.chunk_id: c.score for c in candidates}
            rrf_ranks = {c.chunk_id: i for i, c in enumerate(candidates, start=1)}

            try:
                rerank_results = self.reranker.rerank(query, documents)
            except Exception as exc:
                logger.warning("Reranker failed, using RRF order: %s", exc)
                end_span(output={"status": "fallback", "reason": str(exc)}, level="WARNING")
                return candidates, [], "fallback", str(exc)

            if not rerank_results:
                reason = "reranker_returned_empty_results"
                logger.warning("Reranker returned empty results, using RRF order")
                end_span(output={"status": "fallback", "reason": reason}, level="WARNING")
                return candidates, [], "fallback", reason

            reranked: list[RetrievedChunk] = []
            debug: list[RerankDebugChunk] = []
            for rank, r in enumerate(rerank_results, start=1):
                idx = r.get("index", 0)
                if 0 <= idx < len(candidates):
                    chunk = candidates[idx]
                    relevance_score = r.get("relevance_score", chunk.score)
                    chunk.score = relevance_score
                    chunk.retrieval_path = "rerank"
                    reranked.append(chunk)
                    debug.append(
                        RerankDebugChunk(
                            rank=rank,
                            chunk_id=chunk.chunk_id,
                            title=chunk.title,
                            category=chunk.category,
                            original_index=idx,
                            rrf_rank=rrf_ranks.get(chunk.chunk_id),
                            rrf_score=rrf_scores.get(chunk.chunk_id),
                            relevance_score=relevance_score,
                            content_preview=" ".join(chunk.content.split())[:160],
                        ),
                    )

            if not reranked:
                reason = "reranker_returned_no_valid_indices"
                logger.warning("Reranker returned no valid indices, using RRF order")
                end_span(output={"status": "fallback", "reason": reason}, level="WARNING")
                return candidates, [], "fallback", reason

            end_span(
                output={
                    "status": "ok",
                    "reranked_count": len(reranked),
                    "top_scores": [round(r.score, 4) for r in reranked[:3]],
                    "results": [item.to_dict() for item in debug[:10]],
                },
            )

            return reranked, debug, "ok", None
