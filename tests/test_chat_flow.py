from app.agent.state import session_store
from app.agent.state import ConversationState
from app.config.settings import Settings
from app.agent.purchase_agent import PurchaseDecisionAgent
from app.schemas.chat import ChatRequest
from app.schemas.product_metadata import ProductRecord
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


def test_chat_agent_asks_naturally_when_budget_missing(monkeypatch):
    disable_llm(monkeypatch)
    patch_product_store(monkeypatch)
    agent = PurchaseDecisionAgent()
    response = agent.chat(
        ChatRequest(
            session_id="test-chat-missing",
            message="我想买一台",
            candidate_products=["iphone_15", "vivo_x100"],
        )
    )
    assert response.mode == "chat"
    assert response.recommendation is None
    assert "预算" in response.assistant_message


def test_chat_agent_recommends_when_signal_is_enough(monkeypatch):
    disable_llm(monkeypatch)
    patch_product_store(monkeypatch)
    agent = PurchaseDecisionAgent()
    response = agent.chat(
        ChatRequest(
            session_id="test-chat-recommend",
            message="预算5000，主要拍照和日常用，想用三年，也在意维修",
            candidate_products=["iphone_15", "vivo_x100"],
        )
    )
    assert response.mode == "recommendation"
    assert response.recommendation is not None
    assert response.recommendation.scores
    assert response.answer_source == "fallback"
    assert "used_scoring_guardrail_tool" in response.agent_trace


def test_agent_context_uses_raw_product_records(monkeypatch):
    disable_llm(monkeypatch)
    raw_record = {
        "title": "小米14",
        "price_text": "￥3999",
        "params": {"非固定字段": "卫星通信"},
    }
    state = ConversationState(session_id="raw-context")
    state.candidate_product_records = [ProductRecord(name="小米14", data=raw_record)]

    agent = PurchaseDecisionAgent()

    assert agent._candidate_metadata_payload(state) == [raw_record]
