from __future__ import annotations

from typing import List

from app.multimodal_security.embedding import tokenize
from app.multimodal_security.models import (
    AnswerEvaluation,
    RagasScoreCard,
    RetrievalEvaluation,
    RetrievalHit,
)


class MultimodalRAGEvaluator:
    """
    Two-layer evaluation facade:
    1. Corrective RAG on retrieval quality
    2. Adaptive RAG on answer grounding
    3. RAGAS-like score card for interview explanation
    """

    def evaluate_retrieval(self, query: str, hits: List[RetrievalHit]) -> RetrievalEvaluation:
        if not hits:
            return RetrievalEvaluation(
                context_precision=0.0,
                context_recall=0.0,
                needs_rewrite=True,
                reason="没有命中有效知识单元，需要改写 query 并重新检索。",
            )

        query_terms = set(tokenize(query))
        hit_terms = set(tokenize(" ".join(hit.text for hit in hits)))
        overlap = len(query_terms & hit_terms) / max(len(query_terms), 1)
        precision = round(min(0.95, 0.35 + overlap * 0.6), 4)
        recall = round(min(0.95, 0.4 + len({hit.source for hit in hits}) * 0.12), 4)
        needs_rewrite = precision < 0.55 or recall < 0.55
        reason = "检索结果质量可接受。" if not needs_rewrite else "召回证据不稳定，建议触发 Query 改写。"
        return RetrievalEvaluation(
            context_precision=precision,
            context_recall=recall,
            needs_rewrite=needs_rewrite,
            reason=reason,
        )

    def rewrite_query(self, query: str) -> str:
        return f"{query} 排查步骤 相关截图 告警原因"

    def evaluate_answer(self, question: str, answer: str, hits: List[RetrievalHit]) -> AnswerEvaluation:
        query_terms = set(tokenize(question))
        answer_terms = set(tokenize(answer))
        evidence_terms = set(tokenize(" ".join(hit.text for hit in hits)))
        relevancy = round(min(0.95, 0.4 + len(query_terms & answer_terms) / max(len(query_terms), 1) * 0.5), 4)
        faithfulness = round(min(0.95, 0.4 + len(answer_terms & evidence_terms) / max(len(answer_terms), 1) * 0.5), 4)
        needs_human_review = relevancy < 0.58 or faithfulness < 0.58
        reason = "答案与证据匹配度较高。" if not needs_human_review else "答案与证据对齐度偏弱，建议人工审核。"
        return AnswerEvaluation(
            response_relevancy=relevancy,
            faithfulness=faithfulness,
            needs_human_review=needs_human_review,
            reason=reason,
        )

    def build_ragas_scorecard(self, retrieval_eval: RetrievalEvaluation, answer_eval: AnswerEvaluation) -> RagasScoreCard:
        return RagasScoreCard(
            context_precision=retrieval_eval.context_precision,
            context_recall=retrieval_eval.context_recall,
            response_relevancy=answer_eval.response_relevancy,
            faithfulness=answer_eval.faithfulness,
        )
