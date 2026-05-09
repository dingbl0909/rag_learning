from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AnswerGroundingResult:
    response_relevancy: float
    faithfulness: float
    should_retry: bool
    reason: str


class AdaptiveRAGEvaluator:
    """
    Blueprint for answer-side validation.

    This mirrors the txt statement that answer generation should be checked
    before delivery and retried when grounding is weak.
    """

    def assess_answer(self, question: str, answer: str, evidence_size: int) -> AnswerGroundingResult:
        if not answer.strip():
            return AnswerGroundingResult(0.0, 0.0, True, "Empty answer must be regenerated.")

        relevancy = min(0.95, 0.45 + evidence_size * 0.12)
        faithfulness = min(0.95, 0.42 + evidence_size * 0.1)
        should_retry = relevancy < 0.6 or faithfulness < 0.6
        reason = "Answer is grounded enough to deliver." if not should_retry else "Answer grounding is weak, suggest regeneration."
        return AnswerGroundingResult(relevancy, faithfulness, should_retry, reason)
