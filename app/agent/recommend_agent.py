from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any

from pydantic import Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.agent.prompts import AGENT_TOOL_SYSTEM_PROMPT, SCORING_GUARDRAIL_HINT
from app.agent.schemas import AgentResponse
from app.config.settings import Settings
from app.tools.knowledge_search_tool import KnowledgeSearchTool
from app.tools.web_search_tool import WebSearchTool


@dataclass
class RecommendDeps:
    """Runtime dependencies injected into the agent's tools."""

    web_search_tool: WebSearchTool
    knowledge_search_tool: KnowledgeSearchTool | None = None
    trace: list[str] = field(default_factory=list)


def build_recommend_agent(settings: Settings) -> Agent[RecommendDeps, AgentResponse]:
    """Build the pydantic-ai agent backed by DeepSeek's OpenAI-compatible API.

    Tool schema, the call loop, parameter validation and tool_call wiring are all
    handled by pydantic-ai; we only declare the tool and inject dependencies.
    Returns structured AgentResponse instead of raw str.
    """
    system_prompt = AGENT_TOOL_SYSTEM_PROMPT
    # 打分开关打开时，才把「辅助参考」提示拼进 system prompt；关闭时上下文里根本没有
    # scoring_guardrail_result，也就不提它。
    if settings.scoring_enabled:
        system_prompt = f"{system_prompt}\n\n{SCORING_GUARDRAIL_HINT}"
    agent = Agent(
        _build_model(settings),
        deps_type=RecommendDeps,
        output_type=AgentResponse,
        system_prompt=system_prompt,
    )

    @agent.tool
    def web_search(
        ctx: RunContext[RecommendDeps],
        query: Annotated[
            str,
            Field(description="搜索关键词或自然语言问题，建议包含商品名 + 关注维度（+ 可选平台），例如 'iPhone 15 续航 实际体验 小红书'。"),
        ],
        count: Annotated[
            int,
            Field(description="返回结果数量，1-10，默认 6。", ge=1, le=10),
        ] = 6,
    ) -> dict[str, Any]:
        """联网搜索全网公开信息，获取手机的真实口碑、评测、用户反馈、维修和价格线索。

        当你需要商品原始元信息之外的真实证据时调用我。一轮可以针对不同商品或不同维度发起多个搜索。
        返回若干条网页的标题、来源站点、摘要和链接，不是完整帖子。
        """
        ctx.deps.trace.append("called:web_search")
        return ctx.deps.web_search_tool.run(query=query, count=count)

    @agent.tool
    def knowledge_search(
        ctx: RunContext[RecommendDeps],
        query: Annotated[
            str,
            Field(description="关于产品选购、品牌对比、价格分析、售后政策的自然语言问题，例如 '蓝牙耳机怎么选' 或 '机械键盘轴体对比'。"),
        ],
        category: Annotated[
            str | None,
            Field(description="可选的商品品类过滤，如 '蓝牙耳机'、'机械键盘'、'投影仪'、'显示器'。留空则搜索全部品类。"),
        ] = None,
        top_k: Annotated[
            int,
            Field(description="返回知识条数，1-10，默认 5。", ge=1, le=10),
        ] = 5,
    ) -> dict[str, Any]:
        """搜索私域知识库，获取选购指南、品牌分析、性价比矩阵、售后政策、专家避坑建议等专业领域知识。

        这些知识是结构化经验数据，联网搜索拿不到。当你需要回答选购建议、品牌对比、价格区间分析时调用。
        """
        ctx.deps.trace.append("called:knowledge_search")
        if ctx.deps.knowledge_search_tool is None or not ctx.deps.knowledge_search_tool.enabled:
            return {"status": "disabled", "result_count": 0, "results": []}
        return ctx.deps.knowledge_search_tool.search(query=query, category=category, top_k=top_k)

    return agent


def build_finisher_agent(settings: Settings) -> Agent[None, AgentResponse]:
    """A tool-less agent used to wrap up when the search round limit is hit.

    It replays the partial run's message history and answers from the evidence
    already gathered, instead of discarding everything and falling back.
    Returns structured AgentResponse.
    """
    return Agent(
        _build_model(settings),
        output_type=AgentResponse,
        system_prompt="你是 GizmoGuide。基于已获取的信息给出回复。把给用户看的自然语言回复写在 reply 字段里。",
    )


def _build_model(settings: Settings) -> OpenAIChatModel:
    return OpenAIChatModel(
        settings.deepseek_model,
        provider=OpenAIProvider(
            base_url=settings.deepseek_base_url.rstrip("/"),
            api_key=settings.deepseek_api_key,
        ),
    )
