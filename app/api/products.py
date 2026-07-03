from __future__ import annotations

from fastapi import APIRouter, Query

from app.connectors.redis_product_metadata import get_product_metadata_store
from app.config.settings import get_settings
from app.schemas.product_metadata import ProductSearchResponse

router = APIRouter(tags=["products"])


@router.get("/products", response_model=ProductSearchResponse)
def search_products(
    q: str = Query(default="", description="Phone model search keyword"),
    limit: int | None = Query(default=None, ge=1, le=50),
) -> ProductSearchResponse:
    settings = get_settings()
    resolved_limit = limit or settings.product_search_default_limit
    items = get_product_metadata_store().search(q, resolved_limit)
    return ProductSearchResponse(query=q, total=len(items), items=items)
