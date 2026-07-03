from __future__ import annotations

from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart

from app.agent.recommend_agent import RecommendDeps, build_recommend_agent
from app.config.settings import Settings
from app.tools.web_search_tool import WebSearchTool
from tests.product_store_fakes import patch_product_store


def _settings() -> Settings:
    return Settings(
        deepseek_api_key="sk-test",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-chat",
        llm_timeout_seconds=5,
        bocha_api_key="bocha-test",
        bocha_base_url="https://api.bochaai.com",
        web_search_timeout_seconds=5,
        web_search_cache_ttl_seconds=10,
        agent_max_tool_rounds=4,
    )


def _deps_with_stub(settings: Settings, captured: dict) -> RecommendDeps:
    tool = WebSearchTool(settings)

    def fake_run(query, count=6):
        captured["query"] = query
        captured["count"] = count
        return {
            "status": "ok",
            "query": query,
            "result_count": 1,
            "results": [{"title": "t", "site": "小红书", "url": "http://x", "summary": "续航不错"}],
            "cached": False,
        }

    tool.run = fake_run  # type: ignore[method-assign]
    return RecommendDeps(web_search_tool=tool)


def test_agent_calls_web_search_then_answers():
    settings = _settings()
    agent = build_recommend_agent(settings)
    captured: dict = {}
    deps = _deps_with_stub(settings, captured)

    turns = {"n": 0}

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        turns["n"] += 1
        if turns["n"] == 1:
            return ModelResponse(
                parts=[ToolCallPart(tool_name="web_search", args={"query": "iPhone 15 续航 小红书", "count": 3})]
            )
        return ModelResponse(parts=[TextPart(content='{"reply":"综合小红书反馈，iPhone 15 日常续航够用。"}')])

    with agent.override(model=FunctionModel(model_fn)):
        result = agent.run_sync("帮我对比 iPhone 15 和 vivo X100，主要看续航", deps=deps)

    assert "续航" in result.output.reply
    assert "called:web_search" in deps.trace
    assert captured["query"] == "iPhone 15 续航 小红书"
    assert captured["count"] == 3
    assert turns["n"] == 2


def test_agent_can_answer_without_tool():
    settings = _settings()
    agent = build_recommend_agent(settings)
    captured: dict = {}
    deps = _deps_with_stub(settings, captured)

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content='{"reply":"你先告诉我预算大概多少？"}')])

    with agent.override(model=FunctionModel(model_fn)):
        result = agent.run_sync("帮我看看", deps=deps)

    assert "预算" in result.output.reply
    assert "called:web_search" not in deps.trace
    assert captured == {}


def test_web_search_tool_is_registered_with_schema():
    settings = _settings()
    agent = build_recommend_agent(settings)
    # pydantic-ai 从函数签名 + docstring 自动生成 schema，这里确认工具确实被注册。
    toolset = agent._function_toolset  # noqa: SLF001 - introspection for test only
    assert "web_search" in toolset.tools


def test_agent_finishes_gracefully_on_round_limit(monkeypatch):
    """触顶时应基于已搜到的信息收尾，而不是丢弃证据回退到本地兜底。"""
    import app.agent.purchase_agent as pa
    import app.agent.recommend_agent as ra
    from pydantic_ai import Agent
    from app.agent.schemas import AgentResponse
    from app.agent.purchase_agent import PurchaseDecisionAgent
    from app.schemas.chat import ChatRequest

    settings = _settings()
    monkeypatch.setattr(pa, "get_settings", lambda: settings)
    patch_product_store(monkeypatch)

    # 推荐 agent 的底层模型每轮都要求搜索，必然触顶 request_limit。
    def always_search(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(tool_name="web_search", args={"query": "q"})])

    monkeypatch.setattr(ra, "OpenAIChatModel", lambda *a, **k: FunctionModel(always_search))
    monkeypatch.setattr(ra, "OpenAIProvider", lambda *a, **k: None)

    # 收尾用无工具的 agent，确认它基于历史作答。
    def stub_finisher(_settings_arg):
        return Agent(
            FunctionModel(
                lambda messages, info: ModelResponse(parts=[TextPart(content='{"reply":"基于已查到的信息：推荐 vivo X100。"}')])
            ),
            output_type=AgentResponse,
        )

    monkeypatch.setattr(pa, "build_finisher_agent", stub_finisher)

    agent = PurchaseDecisionAgent()
    agent.web_search_tool.run = lambda query, count=6: {  # type: ignore[method-assign]
        "status": "ok", "query": query, "result_count": 1,
        "results": [{"title": "t", "site": "小红书", "url": "http://x", "summary": "好"}],
        "cached": False,
    }

    resp = agent.chat(
        ChatRequest(session_id="limit-1", message="帮我对比并推荐", candidate_products=["iPhone 15", "vivo X100"])
    )

    assert resp.answer_source == "agent"
    assert "tool_round_limit_reached" in resp.agent_trace
    assert "推荐" in resp.assistant_message
