from __future__ import annotations

from app.schemas.product_metadata import ProductRecord


FAKE_PRODUCTS = [
    ProductRecord(
        name="iPhone 15",
        data={
            "zol_id": "iphone_15",
            "title": "iPhone 15",
            "detail_url": "https://example.com/iphone_15",
            "param_url": "https://example.com/iphone_15/param",
            "price_text": "￥4599",
            "params": {
                "产品型号": "iPhone 15",
                "国内发布时间": "2024年",
                "ROM容量": "128GB",
                "重量": "171g",
            },
        },
    ),
    ProductRecord(
        name="vivo X100",
        data={
            "zol_id": "vivo_x100",
            "title": "vivo X100",
            "detail_url": "https://example.com/vivo_x100",
            "param_url": "https://example.com/vivo_x100/param",
            "price_text": "￥3999",
            "params": {
                "产品型号": "vivo X100",
                "国内发布时间": "2024年",
                "ROM容量": "256GB",
                "重量": "206g",
            },
        },
    ),
    ProductRecord(
        name="Redmi K70",
        data={
            "zol_id": "redmi_k70",
            "title": "Redmi K70",
            "detail_url": "https://example.com/redmi_k70",
            "param_url": "https://example.com/redmi_k70/param",
            "price_text": "￥2499",
            "params": {
                "产品型号": "Redmi K70",
                "国内发布时间": "2024年",
                "ROM容量": "256GB",
                "重量": "209g",
            },
        },
    ),
    ProductRecord(
        name="iPhone 15 Pro",
        data={
            "zol_id": "iphone_15_pro",
            "title": "iPhone 15 Pro",
            "detail_url": "https://example.com/iphone_15_pro",
            "param_url": "https://example.com/iphone_15_pro/param",
            "price_text": "￥6499",
            "params": {
                "产品型号": "iPhone 15 Pro",
                "国内发布时间": "2024年",
                "ROM容量": "128GB",
                "重量": "187g",
            },
        },
    ),
]


class FakeProductMetadataStore:
    def __init__(self) -> None:
        self.products = {product.data["zol_id"]: product for product in FAKE_PRODUCTS}

    def find_products(self, values: list[str]):
        records: list[ProductRecord] = []
        missing: list[str] = []
        seen: set[str] = set()
        for value in values:
            record = self._find_one(value)
            if record is None:
                missing.append(value)
                continue
            if record.name in seen:
                continue
            records.append(record)
            seen.add(record.name)
        return records, missing

    def _find_one(self, value: str) -> ProductRecord | None:
        normalized = _normalize(value)
        for record in FAKE_PRODUCTS:
            aliases = [record.name, str(record.data.get("zol_id") or ""), *record.aliases]
            if any(_normalize(alias) == normalized for alias in aliases):
                return record
        return None


def patch_product_store(monkeypatch) -> FakeProductMetadataStore:
    store = FakeProductMetadataStore()
    monkeypatch.setattr("app.tools.product_tool.get_product_metadata_store", lambda: store)
    return store


def _normalize(value: str) -> str:
    return "".join(str(value).lower().split())
