from __future__ import annotations

from typing import Dict, List

from app.multimodal_security.models import MemoryRecord


class RedisLikeConversationMemory:
    """
    Demo memory layer for:
    - short-term context in Redis
    - long-term distilled memory written back to vector store
    """

    def __init__(self):
        self.short_term: Dict[str, List[str]] = {}
        self.long_term: Dict[str, List[MemoryRecord]] = {}

    def append_turn(self, user_id: str, message: str) -> None:
        self.short_term.setdefault(user_id, []).append(message)
        self.short_term[user_id] = self.short_term[user_id][-4:]

    def build_summary(self, user_id: str) -> str:
        history = self.short_term.get(user_id, [])
        if not history:
            return ""
        return " | ".join(history[-3:])

    def persist_long_term(self, user_id: str, summary: str, tags: List[str]) -> MemoryRecord:
        record = MemoryRecord(
            memory_id=f"{user_id}-memory-{len(self.long_term.get(user_id, [])) + 1}",
            user_id=user_id,
            summary=summary,
            tags=tags,
        )
        self.long_term.setdefault(user_id, []).append(record)
        return record

    def retrieve_long_term(self, user_id: str, limit: int = 2) -> List[MemoryRecord]:
        return self.long_term.get(user_id, [])[-limit:]
