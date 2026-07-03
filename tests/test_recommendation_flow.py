from app.config.settings import Settings
from app.orchestrator.conversation import handle_recommendation
from app.schemas.recommendation import RecommendationRequest
from tests.product_store_fakes import patch_product_store


def disable_llm(monkeypatch):
    monkeypatch.setattr(
        "app.agent.purchase_agent.get_settings",
        lambda: Settings(
            deepseek_api_key=None,
            deepseek_base_url="https://api.deepseek.com",
            deepseek_model="deepseek-chat",
            llm_timeout_seconds=1,
            scoring_enabled=True,
        ),
    )


def test_recommendation_flow_returns_answer(monkeypatch):
    disable_llm(monkeypatch)
    patch_product_store(monkeypatch)
    response = handle_recommendation(
        RecommendationRequest(
            user_message="预算5000，主要拍照和日常用，想用三年，也在意维修",
            candidate_products=["iPhone 15", "vivo X100"],
        )
    )
    assert response.need_clarification is False
    assert response.recommendation is not None
    assert response.recommendation.winner_name in {"iPhone 15", "vivo X100"}
    assert response.recommendation.scores
    assert response.answer is not None
    assert response.answer_source == "fallback"
    assert "llm_disabled" in response.agent_trace
    assert "used_scoring_guardrail_tool" in response.agent_trace


def test_recommendation_missing_info_returns_natural_chat(monkeypatch):
    disable_llm(monkeypatch)
    patch_product_store(monkeypatch)
    response = handle_recommendation(
        RecommendationRequest(
            user_message="帮我对比一下",
            candidate_products=["iPhone 15", "vivo X100"],
        )
    )
    assert response.recommendation is None
    assert response.answer is not None
    assert "预算" in response.answer
    assert response.clarification_questions == []


def test_budget_sensitive_student_prefers_affordable_option(monkeypatch):
    disable_llm(monkeypatch)
    patch_product_store(monkeypatch)
    response = handle_recommendation(
        RecommendationRequest(
            user_message="学生党，预算3000以内，日常和轻度游戏，想用三年，性价比重要",
            candidate_products=["Redmi K70", "iPhone 15 Pro"],
        )
    )
    assert response.need_clarification is False
    assert response.recommendation is not None
    assert response.recommendation.winner_name == "Redmi K70"
