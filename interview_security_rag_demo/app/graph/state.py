from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from app.models import SearchHit


@dataclass
class SecurityRAGGraphState:
    question: str = ""
    route: str = ""
    retrieved_hits: List[SearchHit] = field(default_factory=list)
    rewritten_query: str = ""
    answer: str = ""
    requires_human_review: bool = False
    review_reason: str = ""
