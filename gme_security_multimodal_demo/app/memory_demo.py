from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Tuple

from app.embedding_demo import GMEEmbeddingDemo
from app.models import MemoryItem, SearchHit


class MemoryEngineDemo:
    """
    Redis short-term + Milvus long-term context collection (in-memory mock).

    - Short-term: recent turns + rolling summary
    - Query cache: skip retrieval on repeated questions
    - Long-term: vectorized summaries for semantic recall
    """

    def __init__(self, encoder: Optional[GMEEmbeddingDemo] = None):
        self.encoder = encoder or GMEEmbeddingDemo()
        self.short_term: Dict[str, List[str]] = {}
        self.long_term: Dict[str, List[MemoryItem]] = {}
        self.query_cache: Dict[str, Tuple[List[SearchHit], str]] = {}

    def add_user_turn(self, user_id: str, message: str) -> None:
        self.short_term.setdefault(user_id, []).append(message)
        self.short_term[user_id] = self.short_term[user_id][-6:]

    def summarize_short_term(self, user_id: str) -> str:
        history = self.short_term.get(user_id, [])
        if not history:
            return ""
        return "近期会话：" + " | ".join(history[-3:])

    def cache_key(self, user_id: str, question: str, image_ref: Optional[str]) -> str:
        raw = f"{user_id}|{question.strip()}|{image_ref or ''}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get_cached_hits(self, user_id: str, question: str, image_ref: Optional[str]) -> Optional[Tuple[List[SearchHit], str]]:
        key = self.cache_key(user_id, question, image_ref)
        return self.query_cache.get(key)

    def put_cached_hits(self, user_id: str, question: str, image_ref: Optional[str], hits: List[SearchHit], route: str) -> None:
        key = self.cache_key(user_id, question, image_ref)
        self.query_cache[key] = (hits, route)

    def save_long_term(self, user_id: str, summary: str, tags: List[str]) -> MemoryItem:
        item = MemoryItem(
            memory_id=f"{user_id}-memory-{len(self.long_term.get(user_id, [])) + 1}",
            user_id=user_id,
            summary=summary,
            tags=tags,
            dense_vector=self.encoder.encode_text(summary),
        )
        self.long_term.setdefault(user_id, []).append(item)
        return item

    def search_long_term(self, user_id: str, query: str = "", limit: int = 2) -> List[MemoryItem]:
        items = self.long_term.get(user_id, [])
        if not items:
            return []
        if not query.strip():
            return items[-limit:]
        q_vec = self.encoder.encode_text(query)
        ranked = sorted(
            items,
            key=lambda item: sum(a * b for a, b in zip(q_vec, item.dense_vector)),
            reverse=True,
        )
        return ranked[:limit]
