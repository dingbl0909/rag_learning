from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.config import LOW_CONFIDENCE_THRESHOLD, RISK_KEYWORDS
from app.knowledge_base import KnowledgeBase
from app.models import ChatResponse, EvaluationResult, PendingReview, SearchHit
from app.retrieval import tokenize


def route_question(question: str) -> str:
    if any(keyword in question for keyword in ("告警", "布控", "误报", "联动", "事件")):
        return "alarm"
    if any(keyword in question for keyword in ("Milvus", "Docker", "容器", "部署", "服务", "vllm", "启动")):
        return "deployment"
    return "device"


def evaluate_answer(question: str, hits: List[SearchHit]) -> EvaluationResult:
    if not hits:
        return EvaluationResult(
            confidence=0.0,
            requires_human_review=True,
            reason="未检索到相关证据，建议人工确认后再答复。",
        )

    best_hit = hits[0]
    query_tokens = set(tokenize(question))
    evidence_tokens = set(tokenize(" ".join(hit.content for hit in hits)))
    overlap_ratio = len(query_tokens & evidence_tokens) / max(len(query_tokens), 1)
    score_signal = best_hit.hybrid_score / (best_hit.hybrid_score + 8.0)
    confidence = round(min(0.99, score_signal * 0.45 + overlap_ratio * 0.55), 4)

    if any(keyword in question for keyword in RISK_KEYWORDS):
        return EvaluationResult(
            confidence=confidence,
            requires_human_review=True,
            reason="问题包含高风险操作关键词，建议人工审批后再执行。",
        )

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return EvaluationResult(
            confidence=confidence,
            requires_human_review=True,
            reason="回答置信度偏低，建议人工确认后再输出。",
        )

    return EvaluationResult(
        confidence=confidence,
        requires_human_review=False,
        reason="检索证据充分，可直接返回结果。",
    )


def compose_answer(question: str, route: str, hits: List[SearchHit]) -> str:
    if not hits:
        return (
            f"当前路由到 `{route}` 知识域，但没有命中有效文档。\n\n"
            "建议补充更具体的设备型号、告警名称或部署组件后重试。"
        )

    summary_lines = [
        f"已根据问题路由到 `{route}` 知识域，并从知识库中检索到 {len(hits)} 条高相关证据。",
        "",
        "建议处理步骤：",
    ]

    for index, hit in enumerate(hits, start=1):
        summary_lines.append(f"{index}. {hit.content}")

    summary_lines.extend(
        [
            "",
            "证据来源：",
            *[f"- {hit.source} / {hit.title}" for hit in hits],
            "",
            f"结论：以上建议围绕“{question}”组织，可先按 1 -> 2 -> 3 的顺序排查。",
        ]
    )
    return "\n".join(summary_lines)


@dataclass
class SessionState:
    thread_id: str
    user_id: str
    pending_review: Optional[PendingReview] = None
    history: List[str] = field(default_factory=list)


class InterviewDemoWorkflow:
    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base
        self.sessions: Dict[str, SessionState] = {}

    def _get_session(self, thread_id: str, user_id: str) -> SessionState:
        session = self.sessions.get(thread_id)
        if session is None:
            session = SessionState(thread_id=thread_id, user_id=user_id)
            self.sessions[thread_id] = session
        return session

    def chat(self, thread_id: str, user_id: str, question: str) -> ChatResponse:
        route = route_question(question)
        hits = self.knowledge_base.search(question, domain=route)
        evaluation = evaluate_answer(question, hits)
        answer = compose_answer(question, route, hits)
        session = self._get_session(thread_id, user_id)
        session.history.append(question)

        if evaluation.requires_human_review:
            session.pending_review = PendingReview(
                question=question,
                answer=answer,
                evidence=hits,
                evaluation=evaluation,
            )
            return ChatResponse(
                assistant="系统已生成候选答案，但当前需要人工确认后再继续返回。",
                thread_id=thread_id,
                route=route,
                confidence=evaluation.confidence,
                needs_confirmation=True,
                review_reason=evaluation.reason,
                evidence=hits,
            )

        session.pending_review = None
        return ChatResponse(
            assistant=answer,
            thread_id=thread_id,
            route=route,
            confidence=evaluation.confidence,
            needs_confirmation=False,
            review_reason=evaluation.reason,
            evidence=hits,
        )

    def continue_chat(self, thread_id: str, user_id: str, approve: bool, reviewer_note: Optional[str]) -> ChatResponse:
        session = self._get_session(thread_id, user_id)
        pending = session.pending_review
        if pending is None:
            return ChatResponse(
                assistant="当前没有等待人工确认的内容。",
                thread_id=thread_id,
                route="none",
                confidence=1.0,
                needs_confirmation=False,
                review_reason=None,
                evidence=[],
            )

        session.pending_review = None
        route = route_question(pending.question)
        if approve:
            approved_answer = pending.answer
            if reviewer_note:
                approved_answer += f"\n\n人工审核备注：{reviewer_note}"
            return ChatResponse(
                assistant=approved_answer,
                thread_id=thread_id,
                route=route,
                confidence=pending.evaluation.confidence,
                needs_confirmation=False,
                review_reason="人工已确认，可以继续输出。",
                evidence=pending.evidence,
            )

        regenerated_hits = self.knowledge_base.search(f"{pending.question} 排查 建议", domain=route)
        regenerated_answer = compose_answer(pending.question, route, regenerated_hits)
        if reviewer_note:
            regenerated_answer += f"\n\n人工反馈：{reviewer_note}"
        return ChatResponse(
            assistant=regenerated_answer,
            thread_id=thread_id,
            route=route,
            confidence=min(0.99, pending.evaluation.confidence + 0.08),
            needs_confirmation=False,
            review_reason="人工驳回后触发了二次检索与重新组织回答。",
            evidence=regenerated_hits,
        )
