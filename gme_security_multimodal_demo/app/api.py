from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from app.run_example import run_example
from app.workflow_demo import MultimodalSecurityWorkflowDemo


app = FastAPI(title="GME Security Multimodal Demo", version="0.1.0")

docs_dir = Path(__file__).resolve().parent.parent / "data" / "security_multimodal_docs"
workflow = MultimodalSecurityWorkflowDemo(docs_dir)


@app.get("/")
def health() -> dict:
    return {
        "service": app.title,
        "version": app.version,
        "status": "ok",
        "demo_type": "security_multimodal_rag",
    }


@app.get("/example")
def example() -> dict:
    return run_example()


@app.get("/ask")
def ask(question: str, image_ref: str | None = None, user_id: str = "demo_user") -> dict:
    return workflow.ask(user_id=user_id, question=question, image_ref=image_ref).model_dump()
