from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from redis import Redis

from app.config.settings import Settings, get_settings
from app.schemas.product_metadata import ProductRecord, ProductSearchItem, normalize_search_text

ROOT_DIR = Path(__file__).resolve().parents[2]


class RedisProductMetadataStore:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.redis = Redis.from_url(self.settings.product_redis_url, decode_responses=True)
        configured_path = Path(self.settings.product_metadata_dataset_path)
        self.dataset_path = configured_path if configured_path.is_absolute() else (ROOT_DIR / configured_path).resolve()
        self.key_prefix = "product:metadata"
        self.name_index_key = f"{self.key_prefix}:names"
        self.alias_index_key = f"{self.key_prefix}:aliases"
        self.fingerprint_key = f"{self.key_prefix}:fingerprint"

    def ensure_seeded(self) -> int:
        """按数据集指纹幂等灌库：指纹未变且库非空则跳过，变化或空库则清空重灌。

        指纹是数据集文件内容的 sha256，任何内容变化（记录数、字段值）都会触发重灌，
        保证 Redis 与数据集文件严格一致；文件没变时重启秒过，不做无谓写入。
        """
        self.redis.ping()
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Product metadata dataset not found: {self.dataset_path}")
        raw_bytes = self.dataset_path.read_bytes()
        fingerprint = hashlib.sha256(raw_bytes).hexdigest()
        if self._is_up_to_date(fingerprint):
            return self._seeded_count()
        raw_items = json.loads(raw_bytes.decode("utf-8"))
        self._clear_existing()
        pipe = self.redis.pipeline(transaction=False)
        seeded_names: set[str] = set()
        for raw in raw_items:
            record = ProductRecord.from_raw(raw)
            pipe.set(self._record_key(record.name), json.dumps(record.data, ensure_ascii=False))
            pipe.sadd(self.name_index_key, record.name)
            for alias in record.normalized_aliases:
                pipe.hset(self.alias_index_key, alias, record.name)
            seeded_names.add(record.name)
        pipe.execute()
        self.redis.set(self.fingerprint_key, fingerprint)
        return len(seeded_names)

    def _is_up_to_date(self, fingerprint: str) -> bool:
        """指纹匹配且 name 索引非空时，视为已是最新，可跳过重灌。"""
        if self.redis.get(self.fingerprint_key) != fingerprint:
            return False
        return self.redis.scard(self.name_index_key) > 0

    def _seeded_count(self) -> int:
        return self.redis.scard(self.name_index_key)

    def _clear_existing(self) -> None:
        existing_names = list(self.redis.smembers(self.name_index_key))
        if existing_names:
            self.redis.delete(*existing_names)
        keys = list(self.redis.scan_iter(f"{self.key_prefix}:*"))
        if keys:
            self.redis.delete(*keys)

    def search(self, query: str = "", limit: int | None = None) -> list[ProductSearchItem]:
        limit = limit or self.settings.product_search_default_limit
        normalized_query = normalize_search_text(query)
        items = self._all_records()
        if normalized_query:
            items = [item for item in items if item.matches(normalized_query)]
            items.sort(key=lambda item: self._query_sort_key(item, normalized_query), reverse=True)
        else:
            items.sort(key=lambda item: item.sort_key, reverse=True)
        return [item.to_search_item() for item in items[:limit]]

    def find_products(self, values: list[str]) -> tuple[list[ProductRecord], list[str]]:
        records: list[ProductRecord] = []
        missing: list[str] = []
        seen: set[str] = set()
        for value in values:
            record = self.find_one(value)
            if record is None:
                missing.append(value)
                continue
            if record.name in seen:
                continue
            records.append(record)
            seen.add(record.name)
        return records, missing

    def find_one(self, value: str) -> ProductRecord | None:
        normalized = normalize_search_text(value)
        matched_name = self.redis.hget(self.alias_index_key, normalized)
        if matched_name:
            return self._get_by_name(matched_name)
        matches = [item for item in self._all_records() if item.matches(normalized)]
        matches.sort(key=lambda item: self._query_sort_key(item, normalized), reverse=True)
        return matches[0] if matches else None

    def _record_key(self, name: str) -> str:
        return name

    def _get_by_name(self, name: str) -> ProductRecord | None:
        data = self.redis.get(self._record_key(name))
        return ProductRecord(name=name, data=json.loads(data)) if data else None

    def _all_records(self) -> list[ProductRecord]:
        names = sorted(self.redis.smembers(self.name_index_key))
        if not names:
            return []
        pipe = self.redis.pipeline(transaction=False)
        for name in names:
            pipe.get(self._record_key(name))
        values = pipe.execute()
        records: list[ProductRecord] = []
        for name, value in zip(names, values):
            if value:
                records.append(ProductRecord(name=name, data=json.loads(value)))
        return records

    def _query_sort_key(self, item: ProductRecord, normalized_query: str) -> tuple[int, int, int, str]:
        return (item.relevance_score(normalized_query), *item.sort_key)


@lru_cache(maxsize=1)
def get_product_metadata_store() -> RedisProductMetadataStore:
    return RedisProductMetadataStore(get_settings())
