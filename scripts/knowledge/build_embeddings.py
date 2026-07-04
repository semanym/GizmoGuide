"""离线增量构建知识库 embedding 缓存。

把 ``scripts/data/knowledge/*.json`` 语料里每个 chunk 的向量算好，落到一个
单独的缓存文件（默认 ``scripts/data/knowledge_embeddings.json``）。语料 JSON
保持纯文本，向量单独存放（approach A）。

增量：缓存按 (chunk_id + 文本 sha256) 命中，只对新增或文本变更的 chunk 调
DashScope，其余复用旧向量，避免重复付费。

必须有 DASHSCOPE_API_KEY —— 没有 key 就直接报错退出，绝不写零向量污染缓存。

用法：
    python scripts/knowledge/build_embeddings.py
    python scripts/knowledge/build_embeddings.py --force   # 忽略缓存，全量重算
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许脚本直接运行时找到 app 包
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config.settings import get_settings
from app.embedding.client import EmbeddingClient
from app.knowledge.embedding_cache import EmbeddingCache, embed_text, resolve_path, text_sha256
from app.knowledge.seed import load_seed_knowledge

BATCH_SIZE = 5


def main() -> None:
    ap = argparse.ArgumentParser(description="离线构建知识库 embedding 缓存")
    ap.add_argument("--force", action="store_true", help="忽略现有缓存，全量重算")
    args = ap.parse_args()

    settings = get_settings()
    if not settings.dashscope_api_key:
        print("ERROR: DASHSCOPE_API_KEY 未设置，无法生成 embedding（拒绝写零向量）。")
        sys.exit(1)

    corpus_dir = resolve_path(settings.knowledge_corpus_dir)
    cache_path = resolve_path(settings.knowledge_embeddings_path)

    seed_data = load_seed_knowledge(corpus_dir)
    if not seed_data:
        print(f"ERROR: 语料目录为空或不存在：{corpus_dir}")
        sys.exit(1)

    cache = (
        EmbeddingCache(model=settings.dashscope_embedding_model, dimensions=settings.embedding_dimensions)
        if args.force
        else EmbeddingCache.load(
            cache_path,
            model=settings.dashscope_embedding_model,
            dimensions=settings.embedding_dimensions,
        )
    )

    # 找出需要重算的 chunk（新增或文本变更）
    misses: list[dict] = []
    for item in seed_data:
        sha = text_sha256(embed_text(item))
        if cache.get(item["chunk_id"], sha) is None:
            misses.append(item)

    print(f"语料 {len(seed_data)} 个 chunk；命中缓存 {len(seed_data) - len(misses)}，需重算 {len(misses)}。")

    embedder = EmbeddingClient(
        settings.dashscope_api_key,
        model=settings.dashscope_embedding_model,
        dimensions=settings.embedding_dimensions,
    )

    for i in range(0, len(misses), BATCH_SIZE):
        batch = misses[i : i + BATCH_SIZE]
        texts = [embed_text(item) for item in batch]
        vectors = embedder.embed(texts)
        if len(vectors) != len(batch):
            print(f"ERROR: 第 {i // BATCH_SIZE} 批返回向量数 {len(vectors)} 与请求 {len(batch)} 不符。")
            sys.exit(1)
        for item, vector in zip(batch, vectors):
            if not vector or all(v == 0.0 for v in vector):
                print(f"ERROR: chunk {item['chunk_id']} 得到空/零向量，拒绝写入。")
                sys.exit(1)
            cache.put(item["chunk_id"], text_sha256(embed_text(item)), vector)
        print(f"  已完成 {min(i + BATCH_SIZE, len(misses))}/{len(misses)}")

    # 清掉语料里已不存在的 chunk，避免缓存无限膨胀
    valid_ids = {item["chunk_id"] for item in seed_data}
    stale = [cid for cid in cache.vectors if cid not in valid_ids]
    for cid in stale:
        del cache.vectors[cid]
    if stale:
        print(f"清理 {len(stale)} 个已删除 chunk 的缓存向量。")

    cache.save(cache_path)
    print(f"完成：缓存写入 {cache_path}，共 {len(cache.vectors)} 个向量。")


if __name__ == "__main__":
    main()
