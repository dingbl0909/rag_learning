from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from app.multimodal_security.chunker import ChapterSemanticChunker
from app.multimodal_security.embedding import GMEMultimodalEncoder
from app.multimodal_security.evaluation import MultimodalRAGEvaluator
from app.multimodal_security.index import HybridMultimodalIndex
from app.multimodal_security.memory import RedisLikeConversationMemory
from app.multimodal_security.models import DemoResponse, RetrievalHit
from app.multimodal_security.parser import SecurityMultimodalParser


class MultimodalSecurityRAGDemo:
    """
    A readable end-to-end demo for interview explanation.

    This class mirrors the real project stages:
    parse -> chunk -> embed -> index -> retrieve -> evaluate -> answer -> memory -> human review.
    """

    def __init__(self, docs_dir: Path):
        self.docs_dir = docs_dir
        self.parser = SecurityMultimodalParser()
        self.chunker = ChapterSemanticChunker()
        self.encoder = GMEMultimodalEncoder()
        self.evaluator = MultimodalRAGEvaluator()
        self.memory = RedisLikeConversationMemory()
        self.index: Optional[HybridMultimodalIndex] = None
        self.build_index()

    def build_index(self) -> None:
        assets = self.parser.parse_directory(self.docs_dir)
        units = self.chunker.build_units(assets)
        self.index = HybridMultimodalIndex(units, self.encoder)

    def ask(self, user_id: str, question: str, image_ref: Optional[str] = None) -> DemoResponse:
        self.memory.append_turn(user_id, question)
        route = self._route(question, image_ref)
        long_term_memory = self.memory.retrieve_long_term(user_id)

        assert self.index is not None
        hits = self.index.search(query=question, image_ref=image_ref, domain=route, top_k=4)
        retrieval_eval = self.evaluator.evaluate_retrieval(question, hits)

        if retrieval_eval.needs_rewrite:
            rewritten_query = self.evaluator.rewrite_query(question)
            hits = self.index.search(query=rewritten_query, image_ref=image_ref, domain=route, top_k=4)
            retrieval_eval = self.evaluator.evaluate_retrieval(rewritten_query, hits)

        answer = self._compose_answer(question, route, hits, long_term_memory)
        answer_eval = self.evaluator.evaluate_answer(question, answer, hits)
        ragas_scores = self.evaluator.build_ragas_scorecard(retrieval_eval, answer_eval)

        summary = f"{route} | {question} | {answer[:80]}"
        self.memory.persist_long_term(user_id, summary=summary, tags=[route, "multimodal_rag"])

        return DemoResponse(
            route=route,
            answer=answer,
            retrieval_evaluation=retrieval_eval,
            answer_evaluation=answer_eval,
            ragas_scores=ragas_scores,
            hits=hits,
            used_memory=long_term_memory,
        )

    def _route(self, question: str, image_ref: Optional[str]) -> str:
        if image_ref:
            return "image"
        if any(keyword in question for keyword in ("告警", "时间线", "录像", "布控")):
            return "alarm"
        if any(keyword in question for keyword in ("部署", "Milvus", "容器", "OCR", "向量")):
            return "deployment"
        return "device"

    def _compose_answer(self, question: str, route: str, hits: List[RetrievalHit], memories) -> str:
        if not hits:
            return f"当前问题被路由到 `{route}` 场景，但未找到足够的多模态证据。建议补充设备编号、截图或更明确的告警信息。"

        lines = [
            f"问题已路由到 `{route}` 场景。",
            "系统先完成多模态检索，再结合历史上下文给出建议。",
            "",
            "处理建议：",
        ]
        for idx, hit in enumerate(hits[:3], start=1):
            if hit.image_ref:
                lines.append(f"{idx}. {hit.text}（关联图片：{hit.image_ref}）")
            else:
                lines.append(f"{idx}. {hit.text}")

        if memories:
            lines.extend(["", "历史经验摘要："])
            for memory in memories:
                lines.append(f"- {memory.summary}")

        lines.extend(["", "证据来源："])
        for hit in hits[:3]:
            lines.append(f"- {hit.source} / {hit.title}")

        lines.append("")
        lines.append(f"结论：以上内容围绕“{question}”组织，可用于安防研发、实施或运维人员的快速排查。")
        return "\n".join(lines)
