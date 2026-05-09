from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class RagasScoreCard:
    context_precision: float
    context_recall: float
    response_relevancy: float
    faithfulness: float


class RagasEvaluationLoop:
    """
    Lightweight facade that documents where a real RAGAS loop would sit.
    """

    def evaluate(self, question: str, answer: str, retrieved_contexts: List[str]) -> RagasScoreCard:
        context_factor = min(0.9, 0.25 + len(retrieved_contexts) * 0.18)
        answer_factor = 0.7 if answer else 0.0
        return RagasScoreCard(
            context_precision=round(context_factor, 4),
            context_recall=round(min(0.95, context_factor + 0.05), 4),
            response_relevancy=round(answer_factor, 4),
            faithfulness=round(min(0.92, 0.5 + len(retrieved_contexts) * 0.1), 4),
        )

    def as_dict(self, score_card: RagasScoreCard) -> Dict[str, float]:
        return {
            "context_precision": score_card.context_precision,
            "context_recall": score_card.context_recall,
            "response_relevancy": score_card.response_relevancy,
            "faithfulness": score_card.faithfulness,
        }
