from __future__ import annotations

from typing import Dict, List

from app.models import MemoryItem


class MemoryEngineDemo:
    """
    Demo for:
    - Redis short-term memory
    - long-term context write-back to vector store
    """

    def __init__(self):
        self.short_term: Dict[str, List[str]] = {}
        self.long_term: Dict[str, List[MemoryItem]] = {}

    def add_user_turn(self, user_id: str, message: str) -> None:
        self.short_term.setdefault(user_id, []).append(message)
        self.short_term[user_id] = self.short_term[user_id][-4:]

    def summarize_short_term(self, user_id: str) -> str:
        history = self.short_term.get(user_id, [])
        return " | ".join(history[-3:]) if history else ""

    def save_long_term(self, user_id: str, summary: str, tags: List[str]) -> MemoryItem:
        item = MemoryItem(
            memory_id=f"{user_id}-memory-{len(self.long_term.get(user_id, [])) + 1}",
            user_id=user_id,
            summary=summary,
            tags=tags,
        )
        self.long_term.setdefault(user_id, []).append(item)
        return item

    def search_long_term(self, user_id: str, limit: int = 2) -> List[MemoryItem]:
        return self.long_term.get(user_id, [])[-limit:]
