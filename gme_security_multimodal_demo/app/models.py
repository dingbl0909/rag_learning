from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class ParsedBlock(BaseModel):
    block_id: str
    source: str
    title_path: str
    text: str
    media_refs: List[str] = []


class KnowledgeChunk(BaseModel):
    chunk_id: str
    source: str
    domain: str
    modality: str
    title: str
    text: str
    image_ref: Optional[str] = None
    dense_vector: List[float] = []
    sparse_terms: List[str] = []


class SearchHit(BaseModel):
    chunk_id: str
    source: str
    domain: str
    modality: str
    title: str
    text: str
    image_ref: Optional[str]
    dense_score: float
    sparse_score: float
    hybrid_score: float


class MemoryItem(BaseModel):
    memory_id: str
    user_id: str
    summary: str
    tags: List[str] = []


class RetrievalCheck(BaseModel):
    context_precision: float
    context_recall: float
    needs_rewrite: bool
    reason: str


class AnswerCheck(BaseModel):
    response_relevancy: float
    faithfulness: float
    needs_human_review: bool
    reason: str


class RagasMetrics(BaseModel):
    context_precision: float
    context_recall: float
    response_relevancy: float
    faithfulness: float


class DemoResult(BaseModel):
    route: str
    answer: str
    rewritten_query: Optional[str] = None
    retrieval_check: RetrievalCheck
    answer_check: AnswerCheck
    ragas_metrics: RagasMetrics
    hits: List[SearchHit]
    memory_used: List[MemoryItem] = []
