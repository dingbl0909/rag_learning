from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

from app.evaluation.adaptive_rag import AdaptiveRAGEvaluator
from app.evaluation.corrective_rag import CorrectiveRAGEvaluator
from app.graph.state import SecurityRAGGraphState
from app.workflow import compose_answer, route_question


def route_node(state: SecurityRAGGraphState) -> SecurityRAGGraphState:
    state.route = route_question(state.question)
    return state


def retrieval_quality_node(state: SecurityRAGGraphState) -> Dict[str, object]:
    evaluator = CorrectiveRAGEvaluator()
    result = evaluator.assess_retrieval(state.question, state.retrieved_hits)
    return {"retrieval_quality": asdict(result)}


def rewrite_query_node(state: SecurityRAGGraphState) -> SecurityRAGGraphState:
    evaluator = CorrectiveRAGEvaluator()
    state.rewritten_query = evaluator.rewrite_query(state.question)
    return state


def answer_generation_node(state: SecurityRAGGraphState) -> SecurityRAGGraphState:
    state.answer = compose_answer(state.question, state.route, state.retrieved_hits)
    return state


def answer_quality_node(state: SecurityRAGGraphState) -> Dict[str, object]:
    evaluator = AdaptiveRAGEvaluator()
    result = evaluator.assess_answer(state.question, state.answer, len(state.retrieved_hits))
    return {"answer_quality": asdict(result)}


def human_review_node(state: SecurityRAGGraphState) -> SecurityRAGGraphState:
    state.requires_human_review = True
    state.review_reason = "Triggered by low confidence or high-risk action."
    return state
