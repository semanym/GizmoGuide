from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

NAME_CANDIDATE_FIELDS = ("title", "name", "zol_id")
ALIAS_CANDIDATE_FIELDS = ("title", "name", "zol_id")
PRICE_PATTERNS = (r"(?:￥|¥)\s*(\d+)", r"(\d+)\s*元")
YEAR_PATTERN = r"\b(20[12]\d)\b"
YEAR_MONTH_PATTERN = r"(20[12]\d)\D{0,4}(\d{1,2})"


def normalize_search_text(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def iter_scalar_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        values: list[str] = []
        for item in value.values():
            values.extend(iter_scalar_values(item))
        return values
    if isinstance(value, list):
        values: list[str] = []
        for item in value:
            values.extend(iter_scalar_values(item))
        return values
    if value is None:
        return []
    return [str(value)]


def _first_matching_int(values: list[str], patterns: tuple[str, ...], default: int = 0) -> int:
    for pattern in patterns:
        compiled = re.compile(pattern, re.IGNORECASE)
        for value in values:
            match = compiled.search(value.replace(",", ""))
            if match and match.group(1):
                return int(match.group(1))
    return default


class ProductSearchItem(BaseModel):
    id: str
    name: str
    data: dict[str, Any] = Field(default_factory=dict)


class ProductSearchResponse(BaseModel):
    query: str = ""
    total: int
    items: list[ProductSearchItem]


class ProductRecord(BaseModel):
    name: str
    data: dict[str, Any]

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "ProductRecord":
        name = cls._first_named_value(raw, NAME_CANDIDATE_FIELDS)
        if not name:
            raise ValueError(f"Product record has no usable name field: {raw!r}")
        return cls(name=name, data=dict(raw))

    @staticmethod
    def _first_named_value(raw: dict[str, Any], fields: tuple[str, ...]) -> str:
        for field in fields:
            value = str(raw.get(field) or "").strip()
            if value:
                return value
        return ""

    @property
    def scalar_values(self) -> list[str]:
        return iter_scalar_values(self.data)

    @property
    def aliases(self) -> set[str]:
        aliases = {self.name}
        for field in ALIAS_CANDIDATE_FIELDS:
            value = str(self.data.get(field) or "").strip()
            if value:
                aliases.add(value)
        return aliases

    @property
    def normalized_aliases(self) -> set[str]:
        return {normalize_search_text(alias) for alias in self.aliases if alias}

    def matches(self, normalized_query: str) -> bool:
        normalized_haystack = normalize_search_text(" ".join([self.name, *self.scalar_values]))
        return normalized_query in normalized_haystack or normalized_haystack in normalized_query

    def relevance_score(self, normalized_query: str) -> int:
        normalized_values = [
            normalize_search_text(self.name),
            *[normalize_search_text(value) for value in self.scalar_values],
        ]
        if normalized_values and normalized_values[0] == normalized_query:
            return 4
        if any(value == normalized_query for value in normalized_values):
            return 3
        if any(value.startswith(normalized_query) for value in normalized_values):
            return 2
        return 1

    @property
    def sort_key(self) -> tuple[int, int, str]:
        return (self.release_score, 1 if self.price else 0, self.name)

    @property
    def release_score(self) -> int:
        best = 0
        for value in self.scalar_values:
            year_month = re.search(YEAR_MONTH_PATTERN, value)
            if year_month:
                best = max(best, int(year_month.group(1)) * 100 + int(year_month.group(2)))
                continue
            year = re.search(YEAR_PATTERN, value)
            if year:
                best = max(best, int(year.group(1)) * 100)
        return best

    @property
    def price(self) -> int:
        return _first_matching_int(self.scalar_values, PRICE_PATTERNS)

    def to_search_item(self) -> ProductSearchItem:
        return ProductSearchItem(id=self.name, name=self.name, data=self.data)
