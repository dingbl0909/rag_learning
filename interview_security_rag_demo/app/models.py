from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    source: str
    domain: str
    title: str
    content: str


class SearchHit(BaseModel):
    chunk_id: str
    source: str
    domain: str
    title: str
    content: str
    dense_score: float
    sparse_score: float
    hybrid_score: float


class EvaluationResult(BaseModel):
    confidence: float
    requires_human_review: bool
    reason: str


class PendingReview(BaseModel):
    question: str
    answer: str
    evidence: List[SearchHit]
    evaluation: EvaluationResult


class ChatRequest(BaseModel):
    message: str = Field(..., description="User question")
    thread_id: str = Field(..., description="Conversation thread id")
    user_id: str = Field(default="demo_user", description="Current user id")


class ContinueRequest(BaseModel):
    thread_id: str
    user_id: str = "demo_user"
    approve: bool
    reviewer_note: Optional[str] = None


class ChatResponse(BaseModel):
    assistant: str
    thread_id: str
    route: str
    confidence: float
    needs_confirmation: bool
    review_reason: Optional[str] = None
    evidence: List[SearchHit] = []
