from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    deepseek_api_key: str | None
    deepseek_base_url: str
    deepseek_model: str
    llm_timeout_seconds: float
    bocha_api_key: str | None = None
    bocha_base_url: str = "https://api.bochaai.com"
    web_search_timeout_seconds: float = 20.0
    web_search_cache_ttl_seconds: float = 21600.0
    agent_max_tool_rounds: int = 4
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "http://localhost:3000"
    # RAG / DashScope
    dashscope_api_key: str | None = None
    dashscope_embedding_model: str = "text-embedding-v3"
    dashscope_rerank_model: str = "gte-rerank"
    embedding_dimensions: int = 1024
    rag_database_url: str = "postgresql://langfuse:langfuse@postgres:5432/langfuse"
    rag_recall_top_k: int = 10
    rag_rerank_top_n: int = 5
    # Session / long-term memory
    redis_url: str | None = None
    session_ttl_seconds: int = 86400
    long_term_memory_ttl_seconds: int = 15552000
    # Product metadata store
    product_redis_url: str = "redis://localhost:6379/1"
    product_metadata_dataset_path: str = "scripts/data/zol_specs_raw.json"
    product_search_default_limit: int = 20
    # 规则打分护栏开关：开启时把 ZOL 参数按规则算出的确定性对比分作为锚点塞进上下文；
    # 关闭则完全不调用打分引擎，上下文里不带 scoring_guardrail_result。默认关闭。
    scoring_enabled: bool = False

    @property
    def llm_enabled(self) -> bool:
        return bool(self.deepseek_api_key)

    @property
    def web_search_enabled(self) -> bool:
        return bool(self.bocha_api_key)

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def rag_enabled(self) -> bool:
        return bool(self.dashscope_api_key and self.rag_database_url)


def _env_bool(name: str, default: bool) -> bool:
    """从环境变量读布尔开关；未设置时用默认值。接受 1/true/yes/on（大小写不敏感）。"""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_dotenv(path: Path | None = None) -> None:
    env_path = path or Path.cwd() / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "25")),
        bocha_api_key=os.getenv("BOCHA_API_KEY"),
        bocha_base_url=os.getenv("BOCHA_BASE_URL", "https://api.bochaai.com"),
        web_search_timeout_seconds=float(os.getenv("WEB_SEARCH_TIMEOUT_SECONDS", "20")),
        web_search_cache_ttl_seconds=float(os.getenv("WEB_SEARCH_CACHE_TTL_SECONDS", "21600")),
        agent_max_tool_rounds=int(os.getenv("AGENT_MAX_TOOL_ROUNDS", "4")),
        langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        langfuse_host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        dashscope_embedding_model=os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v3"),
        dashscope_rerank_model=os.getenv("DASHSCOPE_RERANK_MODEL", "gte-rerank"),
        embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "1024")),
        rag_database_url=os.getenv("RAG_DATABASE_URL", "postgresql://langfuse:langfuse@postgres:5432/langfuse"),
        rag_recall_top_k=int(os.getenv("RAG_RECALL_TOP_K", "10")),
        rag_rerank_top_n=int(os.getenv("RAG_RERANK_TOP_N", "5")),
        redis_url=os.getenv("REDIS_URL"),
        session_ttl_seconds=int(os.getenv("SESSION_TTL_SECONDS", "86400")),
        long_term_memory_ttl_seconds=int(os.getenv("LONG_TERM_MEMORY_TTL_SECONDS", "15552000")),
        product_redis_url=os.getenv("PRODUCT_REDIS_URL", "redis://localhost:6379/1"),
        product_metadata_dataset_path=os.getenv("PRODUCT_METADATA_DATASET_PATH", "scripts/data/zol_specs_raw.json"),
        product_search_default_limit=int(os.getenv("PRODUCT_SEARCH_DEFAULT_LIMIT", "20")),
        scoring_enabled=_env_bool("SCORING_ENABLED", False),
    )
