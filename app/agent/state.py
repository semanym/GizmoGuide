from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.config.settings import Settings, get_settings
from app.schemas.long_term_memory import LongTermMemory
from app.schemas.product_metadata import ProductRecord
from app.schemas.user_profile import UserProfile

logger = logging.getLogger(__name__)


@dataclass
class ConversationState:
    session_id: str
    user_id: str | None = None
    candidate_product_records: list[ProductRecord] = field(default_factory=list)
    profile: UserProfile = field(default_factory=UserProfile)
    messages: list[dict[str, str]] = field(default_factory=list)
    agent_messages: list[Any] = field(default_factory=list)


class SessionStore(Protocol):
    def get(self, session_id: str) -> ConversationState:
        ...

    def save(self, state: ConversationState) -> None:
        ...


class LongTermMemoryStore(Protocol):
    def get(self, user_id: str) -> LongTermMemory:
        ...

    def save(self, memory: LongTermMemory) -> None:
        ...


class InMemorySessionStore(SessionStore):
    def __init__(self) -> None:
        self._sessions: dict[str, ConversationState] = {}

    def get(self, session_id: str) -> ConversationState:
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationState(session_id=session_id)
        return self._sessions[session_id]

    def save(self, state: ConversationState) -> None:
        self._sessions[state.session_id] = state

class InMemoryLongTermMemoryStore(LongTermMemoryStore):
    def __init__(self) -> None:
        self._memories: dict[str, LongTermMemory] = {}

    def get(self, user_id: str) -> LongTermMemory:
        if user_id not in self._memories:
            self._memories[user_id] = LongTermMemory(user_id=user_id)
        return self._memories[user_id]

    def save(self, memory: LongTermMemory) -> None:
        self._memories[memory.user_id] = memory


class RedisSessionStore(SessionStore):
    def __init__(self, redis_url: str, *, ttl_seconds: int, key_prefix: str = "gizmoguide:session") -> None:
        self.ttl_seconds = ttl_seconds
        self.key_prefix = key_prefix
        self.client = _build_redis_client(redis_url)

    def get(self, session_id: str) -> ConversationState:
        raw = self.client.get(self._key(session_id))
        if not raw:
            return ConversationState(session_id=session_id)
        try:
            return _decode_conversation_state(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to decode session state for %s: %s", session_id, exc)
            return ConversationState(session_id=session_id)

    def save(self, state: ConversationState) -> None:
        self.client.set(self._key(state.session_id), _encode_conversation_state(state), ex=self.ttl_seconds)

    def _key(self, session_id: str) -> str:
        return f"{self.key_prefix}:{session_id}"


class RedisLongTermMemoryStore(LongTermMemoryStore):
    def __init__(self, redis_url: str, *, ttl_seconds: int, key_prefix: str = "gizmoguide:memory") -> None:
        self.ttl_seconds = ttl_seconds
        self.key_prefix = key_prefix
        self.client = _build_redis_client(redis_url)

    def get(self, user_id: str) -> LongTermMemory:
        raw = self.client.get(self._key(user_id))
        if not raw:
            return LongTermMemory(user_id=user_id)
        try:
            return LongTermMemory.model_validate_json(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to decode long-term memory for %s: %s", user_id, exc)
            return LongTermMemory(user_id=user_id)

    def save(self, memory: LongTermMemory) -> None:
        self.client.set(self._key(memory.user_id), memory.model_dump_json(), ex=self.ttl_seconds)

    def _key(self, user_id: str) -> str:
        return f"{self.key_prefix}:{user_id}"


def build_session_store(settings: Settings) -> SessionStore:
    if not settings.redis_url:
        return InMemorySessionStore()
    try:
        store = RedisSessionStore(settings.redis_url, ttl_seconds=settings.session_ttl_seconds)
        store.client.ping()
        return store
    except Exception as exc:  # noqa: BLE001
        logger.warning("Redis session store unavailable, falling back to in-memory store: %s", exc)
        return InMemorySessionStore()


def build_long_term_memory_store(settings: Settings) -> LongTermMemoryStore:
    if not settings.redis_url:
        return InMemoryLongTermMemoryStore()
    try:
        store = RedisLongTermMemoryStore(settings.redis_url, ttl_seconds=settings.long_term_memory_ttl_seconds)
        store.client.ping()
        return store
    except Exception as exc:  # noqa: BLE001
        logger.warning("Redis memory store unavailable, falling back to in-memory store: %s", exc)
        return InMemoryLongTermMemoryStore()


def _build_redis_client(redis_url: str):
    from redis import Redis

    return Redis.from_url(redis_url, decode_responses=True)


def _encode_conversation_state(state: ConversationState) -> str:
    payload = {
        "session_id": state.session_id,
        "user_id": state.user_id,
        "candidate_product_records": [product.model_dump(mode="json") for product in state.candidate_product_records],
        "profile": state.profile.model_dump(mode="json"),
        "messages": state.messages[-40:],
    }
    return json.dumps(payload, ensure_ascii=False)


def _decode_conversation_state(raw: str) -> ConversationState:
    payload = json.loads(raw)
    return ConversationState(
        session_id=payload["session_id"],
        user_id=payload.get("user_id"),
        candidate_product_records=[ProductRecord.model_validate(item) for item in payload.get("candidate_product_records", [])],
        profile=UserProfile.model_validate(payload.get("profile") or {}),
        messages=list(payload.get("messages") or []),
    )


_settings = get_settings()
session_store = build_session_store(_settings)
long_term_memory_store = build_long_term_memory_store(_settings)
