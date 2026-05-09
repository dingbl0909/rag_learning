from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.models import SearchHit


@dataclass
class RetrievalQuality:
    context_precision: float
    context_recall: float
    should_rewrite_query: bool
    reason: str


class CorrectiveRAGEvaluator:
    """
    Blueprint for retrieval-side validation.

    The txt describes a first evaluation layer that decides whether to keep
    the current retrieval result set or trigger query rewriting and re-retrieval.
    """

    def assess_retrieval(self, query: str, hits: List[SearchHit]) -> RetrievalQuality:
        if not hits:
            return RetrievalQuality(0.0, 0.0, True, "No hits found, must rewrite query.")

        precision = min(0.95, 0.4 + len(hits) * 0.12)
        recall = min(0.95, 0.35 + len({hit.source for hit in hits}) * 0.18)
        should_rewrite = precision < 0.55 or recall < 0.55
        reason = "Retrieval evidence is sufficient." if not should_rewrite else "Retrieval quality is weak, recommend query rewrite."
        return RetrievalQuality(precision, recall, should_rewrite, reason)

    def rewrite_query(self, query: str) -> str:
        return f"{query} 排查步骤 相关配置 常见原因"
