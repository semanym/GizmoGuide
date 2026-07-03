"""Reciprocal Rank Fusion (RRF) for combining multiple ranked retrieval lists."""
from __future__ import annotations

from app.knowledge.debug import RRFDebugChunk
from app.knowledge.models import RetrievedChunk


def reciprocal_rank_fusion(
    *result_lists: list[RetrievedChunk],
    k: int = 60,
    top_n: int = 10,
) -> list[RetrievedChunk]:
    """Merge multiple ranked lists using RRF.

    RRF formula: score(d) = sum(1 / (k + rank_i(d))) for each list i.
    This is rank-based (not score-based), so it works even when different
    retrievers produce incomparable score scales.

    Args:
        result_lists: Two or more ranked lists of RetrievedChunk.
        k: RRF constant (default 60, from Cormack et al. 2009).
        top_n: Maximum number of results to return.

    Returns:
        A single merged list sorted by RRF score descending.
    """
    named_lists = [(f"list_{i}", results) for i, results in enumerate(result_lists, start=1)]
    fused, _debug = reciprocal_rank_fusion_with_debug(named_lists, k=k, top_n=top_n)
    return fused


def reciprocal_rank_fusion_with_debug(
    result_lists: list[tuple[str, list[RetrievedChunk]]],
    k: int = 60,
    top_n: int = 10,
) -> tuple[list[RetrievedChunk], list[RRFDebugChunk]]:
    """Merge named ranked lists using RRF and return per-source contribution details."""
    scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}
    source_ranks: dict[str, dict[str, int]] = {}
    source_scores: dict[str, dict[str, float | None]] = {}

    for source_name, results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            rrf_score = 1.0 / (k + rank)
            cid = chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + rrf_score
            chunk_map[cid] = chunk
            source_ranks.setdefault(cid, {})[source_name] = rank
            source_scores.setdefault(cid, {})[source_name] = round(float(chunk.score), 6)

    sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    fused: list[RetrievedChunk] = []
    debug: list[RRFDebugChunk] = []
    for rank, cid in enumerate(sorted_ids[:top_n], start=1):
        chunk = chunk_map[cid]
        chunk.score = scores[cid]
        chunk.retrieval_path = "rrf"
        fused.append(chunk)
        ranks = source_ranks.get(cid, {})
        debug.append(
            RRFDebugChunk(
                rank=rank,
                chunk_id=cid,
                title=chunk.title,
                category=chunk.category,
                rrf_score=scores[cid],
                sources=sorted(ranks.keys()),
                source_ranks=ranks,
                source_scores=source_scores.get(cid, {}),
                content_preview=" ".join(chunk.content.split())[:160],
            ),
        )

    return fused, debug
