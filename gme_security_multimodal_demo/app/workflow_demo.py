from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.models import DemoResult
from app.workflow_graph import MultimodalSecurityGraph

# Backward-compatible alias
MultimodalSecurityWorkflowDemo = MultimodalSecurityGraph


def build_workflow(docs_dir: Path) -> MultimodalSecurityGraph:
    return MultimodalSecurityGraph(docs_dir)


def ask(
    workflow: MultimodalSecurityGraph,
    user_id: str,
    question: str = "",
    image_ref: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> DemoResult:
    return workflow.ask(user_id=user_id, question=question, image_ref=image_ref, thread_id=thread_id)


def resume(workflow: MultimodalSecurityGraph, thread_id: str, approved: bool = True) -> DemoResult:
    return workflow.resume(thread_id=thread_id, approved=approved)
