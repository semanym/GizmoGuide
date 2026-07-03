"""Dual-path recall: BM25 keyword search + dense vector search."""
from __future__ import annotations

import logging
from typing import Any

import jieba

from app.embedding.client import EmbeddingClient
from app.knowledge.debug import debug_chunks
from app.knowledge.models import RetrievedChunk
from app.knowledge.store import KnowledgeStore
from app.tracing import trace_span

logger = logging.getLogger(__name__)

# Silence jieba's verbose loading logs
jieba.setLogLevel(jieba.logging.INFO)

# Custom domain terms for better Chinese tokenization
_CUSTOM_TERMS = [
    "骨传导", "开放式", "入耳式", "降噪", "蓝牙", "耳机",
    "机械键盘", "键帽", "轴体", "红轴", "茶轴", "青轴", "黑轴",
    "投影仪", "流明", "ANSI", "幕布",
    "显示器", "刷新率", "色域", "色准", "面板",
    "南卡", "韶音", "索尼", "三星", "苹果", "华为",
    "Cherry", "Filco", "Keychron", "燃风",
    "性价比", "售后", "质保", "维修", "防水",
    "续航", "充电", "延迟", "音质", "佩戴",
]
for _term in _CUSTOM_TERMS:
    jieba.add_word(_term)


def tokenize_chinese(text: str) -> str:
    """Tokenize Chinese text using jieba, returning space-separated tokens."""
    tokens = jieba.lcut(text)
    # Filter out whitespace and single-char tokens (too noisy for BM25)
    return " ".join(t.strip() for t in tokens if t.strip() and len(t.strip()) > 0)


def dual_recall(
    store: KnowledgeStore,
    embedder: EmbeddingClient,
    query: str,
    top_k: int = 10,
    category: str | None = None,
) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
    """Execute dual-path recall and return (bm25_results, vector_results).

    Returns two separate ranked lists for RRF fusion.
    """
    with trace_span(
        "dual_recall",
        input_data={"query": query, "top_k": top_k, "category": category},
    ) as (span, end_span):
        # Path A: BM25 keyword search
        tokenized_query = tokenize_chinese(query)
        logger.debug("Tokenized query: %s", tokenized_query)
        bm25_results = store.keyword_search(tokenized_query, top_k=top_k)

        # Path B: Dense vector search
        query_embedding = embedder.embed_query(query)
        vector_results = store.vector_search(query_embedding, top_k=top_k)

        # Optional: category filtering (post-retrieval for simplicity)
        if category:
            bm25_results = [r for r in bm25_results if r.category == category]
            vector_results = [r for r in vector_results if r.category == category]

        end_span(
            output={
                "bm25_count": len(bm25_results),
                "vector_count": len(vector_results),
                "tokenized_query": tokenized_query,
                "bm25_results": [
                    {
                        "rank": item.rank,
                        "chunk_id": item.chunk_id,
                        "title": item.title,
                        "score": item.score,
                        "source": item.source,
                    }
                    for item in debug_chunks(bm25_results, limit=10)
                ],
                "vector_results": [
                    {
                        "rank": item.rank,
                        "chunk_id": item.chunk_id,
                        "title": item.title,
                        "score": item.score,
                        "source": item.source,
                    }
                    for item in debug_chunks(vector_results, limit=10)
                ],
            },
            metadata_extra={
                "bm25_top_ids": [r.chunk_id for r in bm25_results[:3]],
                "vector_top_ids": [r.chunk_id for r in vector_results[:3]],
            },
        )

    return bm25_results, vector_results
