from __future__ import annotations

from typing import List

from app.embedding_demo import tokenize
from app.models import AnswerCheck, QueryMode, RagasMetrics, RetrievalCheck, SearchHit


class EvaluationEngineDemo:
    """
    Corrective RAG (retrieval) + Adaptive RAG (answer) + RAGAS score card.
    """

    MAX_REWRITE_ROUNDS = 2

    def evaluate_retrieval(self, query: str, hits: List[SearchHit], query_mode: QueryMode = "text") -> RetrievalCheck:
        if not hits:
            return RetrievalCheck(
                context_precision=0.0,
                context_recall=0.0,
                needs_rewrite=True,
                reason="未命中相关证据，需要改写 query 并重新检索。",
            )
        query_terms = set(tokenize(query)) if query.strip() else set()
        if query_mode == "image":
            precision = round(min(0.95, 0.5 + hits[0].hybrid_score), 4)
            recall = round(min(0.95, 0.45 + len(hits) * 0.1), 4)
        else:
            hit_terms = set(tokenize(" ".join(hit.text for hit in hits)))
            overlap = len(query_terms & hit_terms) / max(len(query_terms), 1) if query_terms else 0.5
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

    def rewrite_query(self, query: str, route: str) -> str:
        suffix = {
            "alarm": "布控规则 算法阈值 误报排查",
            "deployment": "Milvus 索引 向量维度 OCR联调",
            "snapshot": "设备状态 联动配置 现场抓拍",
            "device": "设备接入 接口状态 排查步骤",
        }.get(route, "排查步骤 相关截图")
        base = query.strip() or "根据图片定位相关知识"
        return f"{base} {suffix}"

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
