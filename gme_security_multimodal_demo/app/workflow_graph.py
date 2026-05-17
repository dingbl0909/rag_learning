from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, TypedDict
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.chunker_demo import ChapterSemanticChunkerDemo
from app.embedding_demo import GMEEmbeddingDemo
from app.evaluation_demo import EvaluationEngineDemo
from app.index_demo import HybridIndexDemo
from app.memory_demo import MemoryEngineDemo
from app.models import (
    AnswerCheck,
    DemoResult,
    MemoryItem,
    QueryMode,
    RagasMetrics,
    RetrievalCheck,
    SearchHit,
)
from app.parser_demo import SecurityDocumentParserDemo


class WorkflowState(TypedDict, total=False):
    user_id: str
    question: str
    image_ref: Optional[str]
    route: str
    query_mode: QueryMode
    short_term_summary: str
    long_term_memory: List[MemoryItem]
    cache_hit: bool
    hits: List[SearchHit]
    retrieval_check: RetrievalCheck
    answer_check: AnswerCheck
    ragas_metrics: RagasMetrics
    rewritten_query: Optional[str]
    answer: str
    rewrite_round: int
    human_review_pending: bool
    human_approved: Optional[bool]
    workflow_steps: List[str]
    thread_id: str


class MultimodalSecurityGraph:
    """LangGraph workflow: route -> memory -> retrieve -> CRAG -> answer -> Adaptive -> HITL -> persist."""

    def __init__(self, docs_dir: Path):
        self.docs_dir = docs_dir
        self.parser = SecurityDocumentParserDemo()
        self.chunker = ChapterSemanticChunkerDemo()
        self.encoder = GMEEmbeddingDemo()
        self.evaluator = EvaluationEngineDemo()
        self.memory = MemoryEngineDemo(encoder=self.encoder)
        self.index: Optional[HybridIndexDemo] = None
        self.checkpointer = MemorySaver()
        self._build_index()
        self.graph = self._compile_graph()

    def _build_index(self) -> None:
        parsed_blocks = self.parser.parse_dir(self.docs_dir)
        chunks = self.chunker.build_chunks(parsed_blocks)
        self.index = HybridIndexDemo(chunks, self.encoder)

    def _step(self, state: WorkflowState, name: str, **updates: Any) -> dict:
        steps = list(state.get("workflow_steps") or [])
        steps.append(name)
        return {"workflow_steps": steps, **updates}

    def _route_domain(self, question: str, image_ref: Optional[str]) -> str:
        if image_ref and not question.strip():
            return "snapshot"
        if image_ref:
            return "snapshot"
        if any(k in question for k in ("告警", "误报", "布控", "时间线")):
            return "alarm"
        if any(k in question for k in ("部署", "Milvus", "OCR", "容器", "向量")):
            return "deployment"
        return "device"

    def _query_mode(self, question: str, image_ref: Optional[str]) -> QueryMode:
        if image_ref and not question.strip():
            return "image"
        if image_ref:
            return "text_image"
        return "text"

    def _compile_graph(self):
        builder = StateGraph(WorkflowState)

        builder.add_node("route_input", self._node_route_input)
        builder.add_node("load_memory", self._node_load_memory)
        builder.add_node("retrieve_kb", self._node_retrieve_kb)
        builder.add_node("evaluate_retrieval", self._node_evaluate_retrieval)
        builder.add_node("rewrite_retrieve", self._node_rewrite_retrieve)
        builder.add_node("generate_answer", self._node_generate_answer)
        builder.add_node("evaluate_answer", self._node_evaluate_answer)
        builder.add_node("human_review", self._node_human_review)
        builder.add_node("persist_memory", self._node_persist_memory)

        builder.add_edge(START, "route_input")
        builder.add_edge("route_input", "load_memory")
        builder.add_conditional_edges(
            "load_memory",
            self._after_memory,
            {"cached": "generate_answer", "retrieve": "retrieve_kb"},
        )
        builder.add_edge("retrieve_kb", "evaluate_retrieval")
        builder.add_conditional_edges(
            "evaluate_retrieval",
            self._after_retrieval_eval,
            {"rewrite": "rewrite_retrieve", "answer": "generate_answer"},
        )
        builder.add_edge("rewrite_retrieve", "evaluate_retrieval")
        builder.add_edge("generate_answer", "evaluate_answer")
        builder.add_conditional_edges(
            "evaluate_answer",
            self._after_answer_eval,
            {"human": "human_review", "done": "persist_memory"},
        )
        builder.add_edge("human_review", "persist_memory")
        builder.add_edge("persist_memory", END)

        return builder.compile(checkpointer=self.checkpointer)

    def _after_memory(self, state: WorkflowState) -> str:
        return "cached" if state.get("cache_hit") else "retrieve"

    def _after_retrieval_eval(self, state: WorkflowState) -> str:
        check = state.get("retrieval_check")
        rounds = state.get("rewrite_round", 0)
        if check and check.needs_rewrite and rounds < EvaluationEngineDemo.MAX_REWRITE_ROUNDS:
            return "rewrite"
        return "answer"

    def _after_answer_eval(self, state: WorkflowState) -> str:
        check = state.get("answer_check")
        if check and check.needs_human_review and state.get("human_approved") is None:
            return "human"
        return "done"

    def _node_route_input(self, state: WorkflowState) -> dict:
        question = state.get("question", "")
        image_ref = state.get("image_ref")
        route = self._route_domain(question, image_ref)
        query_mode = self._query_mode(question, image_ref)
        return self._step(
            state,
            "route_input",
            route=route,
            query_mode=query_mode,
            rewrite_round=state.get("rewrite_round", 0),
        )

    def _node_load_memory(self, state: WorkflowState) -> dict:
        user_id = state["user_id"]
        question = state.get("question", "")
        image_ref = state.get("image_ref")
        self.memory.add_user_turn(user_id, question or f"[image]{image_ref}")
        short_term = self.memory.summarize_short_term(user_id)
        long_term = self.memory.search_long_term(user_id, question)
        cached = self.memory.get_cached_hits(user_id, question, image_ref)
        if cached:
            hits, cached_route = cached
            return self._step(
                state,
                "load_memory",
                short_term_summary=short_term,
                long_term_memory=long_term,
                hits=hits,
                route=cached_route,
                cache_hit=True,
            )
        return self._step(
            state,
            "load_memory",
            short_term_summary=short_term,
            long_term_memory=long_term,
            cache_hit=False,
        )

    def _node_retrieve_kb(self, state: WorkflowState) -> dict:
        assert self.index is not None
        hits = self.index.search(
            query=state.get("question", ""),
            image_ref=state.get("image_ref"),
            query_mode=state.get("query_mode", "text"),
            domain=state.get("route"),
            top_k=4,
        )
        return self._step(state, "retrieve_kb", hits=hits)

    def _node_evaluate_retrieval(self, state: WorkflowState) -> dict:
        check = self.evaluator.evaluate_retrieval(
            state.get("question", ""),
            state.get("hits") or [],
            state.get("query_mode", "text"),
        )
        return self._step(state, "evaluate_retrieval", retrieval_check=check)

    def _node_rewrite_retrieve(self, state: WorkflowState) -> dict:
        assert self.index is not None
        rewritten = self.evaluator.rewrite_query(state.get("question", ""), state.get("route", "device"))
        hits = self.index.search(
            query=rewritten,
            image_ref=state.get("image_ref"),
            query_mode=state.get("query_mode", "text"),
            domain=state.get("route"),
            top_k=4,
        )
        return self._step(
            state,
            "rewrite_retrieve",
            rewritten_query=rewritten,
            hits=hits,
            rewrite_round=state.get("rewrite_round", 0) + 1,
        )

    def _node_generate_answer(self, state: WorkflowState) -> dict:
        answer = self._compose_answer(
            state.get("question", ""),
            state.get("route", "device"),
            state.get("query_mode", "text"),
            state.get("hits") or [],
            state.get("long_term_memory") or [],
            state.get("short_term_summary", ""),
            state.get("cache_hit", False),
        )
        return self._step(state, "generate_answer", answer=answer)

    def _node_evaluate_answer(self, state: WorkflowState) -> dict:
        check = self.evaluator.evaluate_answer(
            state.get("question", ""),
            state.get("answer", ""),
            state.get("hits") or [],
        )
        ragas = self.evaluator.build_ragas_metrics(state["retrieval_check"], check)
        return self._step(state, "evaluate_answer", answer_check=check, ragas_metrics=ragas)

    def _node_human_review(self, state: WorkflowState) -> dict:
        if state.get("human_approved") is not None:
            return self._step(
                state,
                "human_review",
                human_review_pending=False,
                human_approved=state["human_approved"],
            )
        payload = {
            "question": state.get("question"),
            "answer": state.get("answer"),
            "reason": state["answer_check"].reason,
        }
        decision = interrupt(payload)
        approved = bool(decision.get("approved", False)) if isinstance(decision, dict) else bool(decision)
        if not approved:
            answer = (state.get("answer") or "") + "\n\n[人工驳回] 请补充设备截图、告警时间线或更明确的部署上下文后重试。"
        else:
            answer = (state.get("answer") or "") + "\n\n[人工确认] 以上建议可作为处置参考。"
        return self._step(
            state,
            "human_review",
            human_approved=approved,
            human_review_pending=False,
            answer=answer,
        )

    def _node_persist_memory(self, state: WorkflowState) -> dict:
        user_id = state["user_id"]
        if not state.get("cache_hit") and state.get("hits"):
            self.memory.put_cached_hits(
                user_id,
                state.get("question", ""),
                state.get("image_ref"),
                state.get("hits") or [],
                state.get("route", "device"),
            )
        self.memory.save_long_term(
            user_id=user_id,
            summary=f"{state.get('route')} | {state.get('question', '[image]')} | {(state.get('answer') or '')[:80]}",
            tags=[state.get("route", "device"), "multimodal_rag", "security"],
        )
        return self._step(state, "persist_memory")

    def _compose_answer(
        self,
        question: str,
        route: str,
        query_mode: QueryMode,
        hits: List[SearchHit],
        memory_used: List[MemoryItem],
        short_term_summary: str,
        cache_hit: bool,
    ) -> str:
        mode_note = {"text": "文本检索", "image": "纯图检索", "text_image": "图文联合检索"}[query_mode]
        if not hits:
            return (
                f"当前问题已路由到 `{route}` 场景（{mode_note}），"
                "但未检索到足够证据，建议补充截图或更明确的故障描述。"
            )
        lines = [
            f"当前问题已路由到 `{route}` 场景，检索模式：{mode_note}。",
        ]
        if cache_hit:
            lines.append("（命中会话缓存，跳过重复向量检索）")
        lines.extend(["", "建议："])
        for idx, hit in enumerate(hits[:3], start=1):
            if hit.modality == "image":
                media_note = f"（图片证据：{hit.image_ref}）"
            elif hit.image_ref:
                media_note = f"（图文证据：{hit.image_ref}）"
            else:
                media_note = ""
            lines.append(f"{idx}. {hit.text}{media_note}")
        if short_term_summary:
            lines.extend(["", "短期会话摘要：", f"- {short_term_summary}"])
        if memory_used:
            lines.extend(["", "长期语义记忆："])
            for item in memory_used:
                lines.append(f"- {item.summary}")
        lines.extend(["", "证据来源："])
        for hit in hits[:3]:
            lines.append(f"- {hit.source} / {hit.title} [{hit.modality}]")
        q_display = question or "（仅图片输入）"
        lines.append("")
        lines.append(f"结论：以上建议围绕「{q_display}」组织，适用于安防研发、实施或运维排查。")
        return "\n".join(lines)

    def ask(
        self,
        user_id: str,
        question: str = "",
        image_ref: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> DemoResult:
        tid = thread_id or str(uuid4())
        initial: WorkflowState = {
            "user_id": user_id,
            "question": question,
            "image_ref": image_ref,
            "workflow_steps": [],
            "rewrite_round": 0,
            "human_approved": None,
            "thread_id": tid,
        }
        config = {"configurable": {"thread_id": tid}}
        final = self.graph.invoke(initial, config=config)
        interrupted = bool(self.graph.get_state(config).next)
        return self._to_result(final, tid, interrupted)

    def resume(self, thread_id: str, approved: bool = True, human_note: str = "") -> DemoResult:
        config = {"configurable": {"thread_id": thread_id}}
        payload: Any = {"approved": approved}
        if human_note:
            payload["note"] = human_note
        final = self.graph.invoke(Command(resume=payload), config=config)
        interrupted = bool(self.graph.get_state(config).next)
        return self._to_result(final, thread_id, interrupted)

    def _to_result(self, state: WorkflowState, thread_id: str, interrupted: bool) -> DemoResult:
        retrieval_check = state.get("retrieval_check") or RetrievalCheck(
            context_precision=0.0,
            context_recall=0.0,
            needs_rewrite=False,
            reason="",
        )
        answer_check = state.get("answer_check") or AnswerCheck(
            response_relevancy=0.0,
            faithfulness=0.0,
            needs_human_review=False,
            reason="",
        )
        ragas = state.get("ragas_metrics") or self.evaluator.build_ragas_metrics(retrieval_check, answer_check)
        pending = interrupted or (
            answer_check.needs_human_review and state.get("human_approved") is None
        )
        return DemoResult(
            route=state.get("route", "device"),
            query_mode=state.get("query_mode", "text"),
            answer=state.get("answer", ""),
            rewritten_query=state.get("rewritten_query"),
            retrieval_check=retrieval_check,
            answer_check=answer_check,
            ragas_metrics=ragas,
            hits=state.get("hits") or [],
            memory_used=state.get("long_term_memory") or [],
            short_term_summary=state.get("short_term_summary", ""),
            cache_hit=bool(state.get("cache_hit")),
            human_review_pending=pending,
            thread_id=thread_id,
            workflow_steps=state.get("workflow_steps") or [],
        )
