from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.product_metadata import ProductRecord
from app.schemas.user_profile import UserProfile


class RecommendationRequest(BaseModel):
    user_id: str | None = None
    user_message: str
    candidate_products: list[str] = Field(min_length=2, max_length=5)
    profile: UserProfile | None = None


class ProductScore(BaseModel):
    product_id: str
    product_name: str
    total_score: float
    dimension_scores: dict[str, float]
    penalties: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)


class RecommendationResult(BaseModel):
    winner_id: str
    winner_name: str
    confidence: float = Field(ge=0, le=1)
    scores: list[ProductScore] = Field(default_factory=list)
    key_reasons: list[str]
    risks: list[str]
    reversal_conditions: list[str]
    missing_information: list[str]


class ClarificationQuestion(BaseModel):
    field: str
    question: str


class RecommendationResponse(BaseModel):
    need_clarification: bool
    clarification_questions: list[ClarificationQuestion] = Field(default_factory=list)
    user_profile: UserProfile
    products: list[ProductRecord] = Field(default_factory=list)
    recommendation: RecommendationResult | None = None
    answer: str | None = None
    answer_source: str = "fallback"
    agent_trace: list[str] = Field(default_factory=list)
