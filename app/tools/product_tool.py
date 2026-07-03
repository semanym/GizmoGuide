from __future__ import annotations

from typing import Protocol

from app.connectors.redis_product_metadata import get_product_metadata_store
from app.schemas.product_metadata import ProductRecord


class ProductStore(Protocol):
    def find_products(self, values: list[str]) -> tuple[list[ProductRecord], list[str]]:
        ...


class ProductTool:
    name = "product_spec_tool"

    def __init__(self, store: ProductStore | None = None):
        self.store = store or get_product_metadata_store()

    def get_products(self, product_ids_or_names: list[str]) -> tuple[list[ProductRecord], list[str]]:
        return self.store.find_products(product_ids_or_names)
