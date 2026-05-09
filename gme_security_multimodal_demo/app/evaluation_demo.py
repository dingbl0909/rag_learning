from __future__ import annotations

from typing import List

from app.embedding_demo import tokenize
from app.models import AnswerCheck, RagasMetrics, RetrievalCheck, SearchHit


class EvaluationEngineDemo:
    """
    Demo for:
    - Corrective RAG retrieval evaluation
    - Adaptive RAG answer evaluation
    - RAGAS score card
    """

    def evaluate_retrieval(self, query: str, hits: List[SearchHit]) -> RetrievalCheck:
        if not hits:
            return RetrievalCheck(
                context_precision=0.0,
                context_recall=0.0,
                needs_rewrite=True,
                reason="未命中相关证据，需要改写 query 并重新检索。",
            )
        query_terms = set(tokenize(query))
        hit_terms = set(tokenize(" ".join(hit.text for hit in hits)))
        overlap = len(query_terms & hit_terms) / max(len(query_terms), 1)
        precision = round(min(0.95, 0.35 + overlap * 0.6), 4)
        recall = round(min(0.95, 0.4 + len({hit.source for hit in hits}) * 0.12), 4)
        needs_rewrite = precision < 0.55 or recall < 0.55
        reason = "检索证据足够支撑回答。" if not needs_rewrite else "召回质量偏弱，建议执行 Query 改写。"
        return RetrievalCheck(
            context_precision=precision,
            context_recall=recall,
            needs_rewrite=needs_rewrite,
            reason=reason,
        )

    def rewrite_query(self, query: str) -> str:
        return f"{query} 排查步骤 设备状态 相关截图"

    def evaluate_answer(self, question: str, answer: str, hits: List[SearchHit]) -> AnswerCheck:
        q_terms = set(tokenize(question))
        a_terms = set(tokenize(answer))
        e_terms = set(tokenize(" ".join(hit.text for hit in hits)))
        relevancy = round(min(0.95, 0.4 + len(q_terms & a_terms) / max(len(q_terms), 1) * 0.5), 4)
        faithfulness = round(min(0.95, 0.4 + len(a_terms & e_terms) / max(len(a_terms), 1) * 0.5), 4)
        needs_human_review = relevancy < 0.58 or faithfulness < 0.58
        reason = "答案与证据对齐度较高。" if not needs_human_review else "答案存在不稳定风险，建议人工审核。"
        return AnswerCheck(
            response_relevancy=relevancy,
            faithfulness=faithfulness,
            needs_human_review=needs_human_review,
            reason=reason,
        )

    def build_ragas_metrics(self, retrieval_check: RetrievalCheck, answer_check: AnswerCheck) -> RagasMetrics:
        return RagasMetrics(
            context_precision=retrieval_check.context_precision,
            context_recall=retrieval_check.context_recall,
            response_relevancy=answer_check.response_relevancy,
            faithfulness=answer_check.faithfulness,
        )
