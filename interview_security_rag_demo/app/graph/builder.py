from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class WorkflowBlueprint:
    nodes: List[str]
    edges: List[Tuple[str, str]]
    interrupt_before: List[str]
    notes: List[str]


def build_langgraph_blueprint() -> WorkflowBlueprint:
    return WorkflowBlueprint(
        nodes=[
            "route_node",
            "retriever_node",
            "retrieval_quality_node",
            "rewrite_query_node",
            "answer_generation_node",
            "answer_quality_node",
            "human_review_node",
        ],
        edges=[
            ("route_node", "retriever_node"),
            ("retriever_node", "retrieval_quality_node"),
            ("retrieval_quality_node", "rewrite_query_node"),
            ("retrieval_quality_node", "answer_generation_node"),
            ("rewrite_query_node", "retriever_node"),
            ("answer_generation_node", "answer_quality_node"),
            ("answer_quality_node", "human_review_node"),
        ],
        interrupt_before=["human_review_node"],
        notes=[
            "Use LangGraph StateGraph in production.",
            "The current demo keeps the workflow runnable via plain Python while exposing the graph blueprint explicitly.",
        ],
    )


def blueprint_as_dict() -> Dict[str, object]:
    blueprint = build_langgraph_blueprint()
    return {
        "nodes": blueprint.nodes,
        "edges": blueprint.edges,
        "interrupt_before": blueprint.interrupt_before,
        "notes": blueprint.notes,
    }
