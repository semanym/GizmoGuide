import json
from pathlib import Path

from app.config.settings import Settings
from app.connectors.redis_product_metadata import RedisProductMetadataStore


class FakePipeline:
    def __init__(self, redis):
        self.redis = redis
        self.calls = []

    def set(self, key, value):
        self.calls.append(("set", key, value))
        return self

    def sadd(self, key, value):
        self.calls.append(("sadd", key, value))
        return self

    def get(self, key):
        self.calls.append(("get", key))
        return self

    def hset(self, key, field, value):
        self.calls.append(("hset", key, field, value))
        return self

    def execute(self):
        results = []
        for call in self.calls:
            if call[0] == "set":
                _, key, value = call
                self.redis.data[key] = value
                results.append(True)
            elif call[0] == "sadd":
                _, key, value = call
                self.redis.sets.setdefault(key, set()).add(value)
                results.append(1)
            elif call[0] == "get":
                _, key = call
                results.append(self.redis.data.get(key))
            elif call[0] == "hset":
                _, key, field, value = call
                self.redis.hashes.setdefault(key, {})[field] = value
                results.append(1)
        return results


class FakeRedis:
    def __init__(self):
        self.data = {}
        self.sets = {}
        self.hashes = {}

    def ping(self):
        return True

    def scard(self, key):
        return len(self.sets.get(key, set()))

    def smembers(self, key):
        return set(self.sets.get(key, set()))

    def get(self, key):
        return self.data.get(key)

    def hget(self, key, field):
        return self.hashes.get(key, {}).get(field)

    def scan_iter(self, pattern):
        prefix = pattern.rstrip("*")
        for key in [*self.data.keys(), *self.sets.keys(), *self.hashes.keys()]:
            if key.startswith(prefix):
                yield key

    def delete(self, *keys):
        for key in keys:
            self.data.pop(key, None)
            self.sets.pop(key, None)
            self.hashes.pop(key, None)
        return len(keys)

    def pipeline(self, transaction=False):
        return FakePipeline(self)


def _settings(dataset_path: Path) -> Settings:
    return Settings(
        deepseek_api_key=None,
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-chat",
        llm_timeout_seconds=1,
        product_redis_url="redis://localhost:6379/1",
        product_metadata_dataset_path=str(dataset_path),
    )


def test_redis_product_metadata_seed_search_and_lookup(tmp_path, monkeypatch):
    dataset = tmp_path / "zol_specs.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "zol_id": "1",
                    "title": "小米14",
                    "detail_url": "https://example.com/1",
                    "param_url": "https://example.com/1/param",
                    "price_text": "￥3999",
                    "params": {
                        "国内发布时间": "2024年10月",
                        "CPU型号": "骁龙 8 Gen3",
                        "ROM容量": "256GB",
                        "电池容量": "4610mAh",
                        "屏幕刷新率": "120Hz",
                        "重量": "193g",
                        "非固定字段": "卫星通信",
                    },
                },
                {
                    "zol_id": "2",
                    "title": "vivo X100",
                    "detail_url": "https://example.com/2",
                    "param_url": "https://example.com/2/param",
                    "price_text": "￥3999",
                    "params": {"国内发布时间": "2023年11月", "CPU型号": "天玑 9300", "ROM容量": "256GB"},
                },
                {
                    "zol_id": "3",
                    "title": "vivo X100 Ultra",
                    "detail_url": "https://example.com/3",
                    "param_url": "https://example.com/3/param",
                    "price_text": "￥5999",
                    "params": {"国内发布时间": "2024年5月", "CPU型号": "骁龙 8 Gen3", "ROM容量": "256GB"},
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    fake_redis = FakeRedis()
    monkeypatch.setattr("app.connectors.redis_product_metadata.Redis.from_url", lambda *a, **k: fake_redis)

    store = RedisProductMetadataStore(_settings(dataset))
    assert store.ensure_seeded() == 3
    assert json.loads(fake_redis.data["小米14"])["params"]["非固定字段"] == "卫星通信"
    assert not any(key.startswith("product:metadata:item:") for key in fake_redis.data)

    results = store.search("小米", limit=5)
    assert len(results) == 1
    assert results[0].id == "小米14"
    assert results[0].name == "小米14"
    assert results[0].data["params"]["非固定字段"] == "卫星通信"

    flexible_results = store.search("卫星通信", limit=5)
    assert [item.name for item in flexible_results] == ["小米14"]

    vivo_results = store.search("vivo X100", limit=5)
    assert [item.name for item in vivo_results[:2]] == ["vivo X100", "vivo X100 Ultra"]

    metadata, missing = store.find_products(["小米14", "vivo X100"])
    assert missing == []
    assert [item.name for item in metadata] == ["小米14", "vivo X100"]
    assert [item.data["zol_id"] for item in metadata] == ["1", "2"]
