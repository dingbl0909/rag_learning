from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI

from app.architecture_overview import build_architecture_overview
from app.graph.builder import blueprint_as_dict
from app.knowledge_base import KnowledgeBase
from app.multimodal_security.demo_runner import run_example
from app.models import ChatRequest, ChatResponse, ContinueRequest
from app.vectorstore.milvus_cluster import build_cluster_blueprint
from app.vectorstore.milvus_schema import (
    build_context_collection_blueprint,
    build_knowledge_collection_blueprint,
)
from app.workflow import InterviewDemoWorkflow


app = FastAPI(title="Interview Security RAG Demo", version="0.1.0")

knowledge_base = KnowledgeBase.build()
workflow = InterviewDemoWorkflow(knowledge_base)


@app.get("/")
def health() -> dict:
    return {
        "service": app.title,
        "version": app.version,
        "documents": len(knowledge_base.chunks),
        "status": "ok",
    }


@app.get("/architecture")
def architecture() -> dict:
    return build_architecture_overview()


@app.get("/architecture/langgraph")
def architecture_langgraph() -> dict:
    return blueprint_as_dict()


@app.get("/architecture/milvus")
def architecture_milvus() -> dict:
    return {
        "cluster": asdict(build_cluster_blueprint()),
        "knowledge_collection": build_knowledge_collection_blueprint().to_dict(),
        "context_collection": build_context_collection_blueprint().to_dict(),
    }


@app.get("/multimodal-security/example")
def multimodal_security_example() -> dict:
    return run_example()


@app.post("/rebuild-index")
def rebuild_index() -> dict:
    global knowledge_base, workflow
    knowledge_base = KnowledgeBase.build()
    workflow = InterviewDemoWorkflow(knowledge_base)
    return {"status": "rebuilt", "documents": len(knowledge_base.chunks)}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    return workflow.chat(thread_id=req.thread_id, user_id=req.user_id, question=req.message)


@app.post("/chat/continue", response_model=ChatResponse)
def continue_chat(req: ContinueRequest) -> ChatResponse:
    return workflow.continue_chat(
        thread_id=req.thread_id,
        user_id=req.user_id,
        approve=req.approve,
        reviewer_note=req.reviewer_note,
    )
