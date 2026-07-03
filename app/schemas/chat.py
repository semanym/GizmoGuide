from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.product_metadata import ProductRecord
from app.schemas.long_term_memory import LongTermMemory
from app.schemas.recommendation import RecommendationResult
from app.schemas.user_profile import UserProfile


class ChatRequest(BaseModel):
    session_id: str = "default"
    user_id: str | None = None
    message: str
    profile: UserProfile | None = None
    candidate_products: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    session_id: str
    user_id: str
    mode: Literal["chat", "recommendation"]
    assistant_message: str
    user_profile: UserProfile
    long_term_memory: LongTermMemory
    products: list[ProductRecord] = Field(default_factory=list)
    recommendation: RecommendationResult | None = None
    answer_source: str = "fallback"
    agent_trace: list[str] = Field(default_factory=list)


class ChatStreamEvent(BaseModel):
    event: Literal[
        "status",
        "memory",
        "tool_start",
        "tool_result",
        "scoring",
        "final",
        "error",
    ]
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
