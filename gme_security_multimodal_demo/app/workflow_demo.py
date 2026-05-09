from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from app.chunker_demo import ChapterSemanticChunkerDemo
from app.embedding_demo import GMEEmbeddingDemo
from app.evaluation_demo import EvaluationEngineDemo
from app.index_demo import HybridIndexDemo
from app.memory_demo import MemoryEngineDemo
from app.models import DemoResult, SearchHit
from app.parser_demo import SecurityDocumentParserDemo


class MultimodalSecurityWorkflowDemo:
    """
    End-to-end demo for understanding the project:
    parse -> chunk -> embed -> hybrid retrieve -> memory -> evaluate -> answer -> human review hint
    """

    def __init__(self, docs_dir: Path):
        self.docs_dir = docs_dir
        self.parser = SecurityDocumentParserDemo()
        self.chunker = ChapterSemanticChunkerDemo()
        self.encoder = GMEEmbeddingDemo()
        self.evaluator = EvaluationEngineDemo()
        self.memory = MemoryEngineDemo()
        self.index: Optional[HybridIndexDemo] = None
        self.build_index()

    def build_index(self) -> None:
        parsed_blocks = self.parser.parse_dir(self.docs_dir)
        chunks = self.chunker.build_chunks(parsed_blocks)
        self.index = HybridIndexDemo(chunks, self.encoder)

    def ask(self, user_id: str, question: str, image_ref: Optional[str] = None) -> DemoResult:
        self.memory.add_user_turn(user_id, question)
        route = self._route(question, image_ref)
        long_term = self.memory.search_long_term(user_id)

        assert self.index is not None
        hits = self.index.search(question, image_ref=image_ref, domain=route, top_k=4)
        retrieval_check = self.evaluator.evaluate_retrieval(question, hits)

        rewritten_query = None
        if retrieval_check.needs_rewrite:
            rewritten_query = self.evaluator.rewrite_query(question)
            hits = self.index.search(rewritten_query, image_ref=image_ref, domain=route, top_k=4)
            retrieval_check = self.evaluator.evaluate_retrieval(rewritten_query, hits)

        answer = self._compose_answer(question, route, hits, long_term)
        answer_check = self.evaluator.evaluate_answer(question, answer, hits)
        ragas_metrics = self.evaluator.build_ragas_metrics(retrieval_check, answer_check)

        self.memory.save_long_term(
            user_id=user_id,
            summary=f"{route} | {question} | {answer[:80]}",
            tags=[route, "multimodal_rag", "security"],
        )

        return DemoResult(
            route=route,
            answer=answer,
            rewritten_query=rewritten_query,
            retrieval_check=retrieval_check,
            answer_check=answer_check,
            ragas_metrics=ragas_metrics,
            hits=hits,
            memory_used=long_term,
        )

    def _route(self, question: str, image_ref: Optional[str]) -> str:
        if image_ref:
            return "snapshot"
        if any(keyword in question for keyword in ("告警", "误报", "布控", "时间线")):
            return "alarm"
        if any(keyword in question for keyword in ("部署", "Milvus", "OCR", "容器", "向量")):
            return "deployment"
        return "device"

    def _compose_answer(self, question: str, route: str, hits: List[SearchHit], memory_used) -> str:
        if not hits:
            return f"当前问题被路由到 `{route}` 场景，但没有找到足够的文本或图片证据，建议补充设备截图、告警时间线或更明确的部署信息。"

        lines = [
            f"当前问题已路由到 `{route}` 场景。",
            "系统先完成多模态检索，再基于证据组织回答。",
            "",
            "建议：",
        ]
        for idx, hit in enumerate(hits[:3], start=1):
            media_note = f"（关联图片或表格：{hit.image_ref}）" if hit.image_ref else ""
            lines.append(f"{idx}. {hit.text}{media_note}")

        if memory_used:
            lines.extend(["", "历史上下文："])
            for item in memory_used:
                lines.append(f"- {item.summary}")

        lines.extend(["", "证据来源："])
        for hit in hits[:3]:
            lines.append(f"- {hit.source} / {hit.title}")

        lines.append("")
        lines.append(f"结论：以上建议围绕“{question}”组织，适用于安防研发、实施或运维人员的排查与处置。")
        return "\n".join(lines)
