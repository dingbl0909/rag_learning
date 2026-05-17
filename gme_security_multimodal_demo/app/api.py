from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from app.run_example import run_all_examples
from app.workflow_graph import MultimodalSecurityGraph

app = FastAPI(title="GME Security Multimodal Demo", version="0.2.0")

docs_dir = Path(__file__).resolve().parent.parent / "data" / "security_multimodal_docs"
workflow = MultimodalSecurityGraph(docs_dir)


@app.get("/")
def health() -> dict:
    return {
        "service": app.title,
        "version": app.version,
        "status": "ok",
        "demo_type": "security_multimodal_rag",
        "workflow": "langgraph",
    }


@app.get("/example")
def example() -> dict:
    return run_all_examples()


@app.get("/ask")
def ask(
    question: str = "",
    image_ref: str | None = None,
    user_id: str = "demo_user",
    thread_id: str | None = None,
) -> dict:
    if not question.strip() and not image_ref:
        raise HTTPException(status_code=400, detail="question 或 image_ref 至少提供一个")
    result = workflow.ask(
        user_id=user_id,
        question=question,
        image_ref=image_ref,
        thread_id=thread_id,
    )
    return result.model_dump()


@app.get("/resume")
def resume(thread_id: str, approved: bool = True, human_note: str = "") -> dict:
    try:
        result = workflow.resume(thread_id=thread_id, approved=approved, human_note=human_note)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"无法恢复线程 {thread_id}: {exc}") from exc
    return result.model_dump()
