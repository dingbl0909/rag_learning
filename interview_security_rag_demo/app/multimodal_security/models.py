from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class ParsedAsset(BaseModel):
    asset_id: str
    source: str
    asset_type: str
    title_path: str
    content: str
    related_media: List[str] = []


class KnowledgeUnit(BaseModel):
    unit_id: str
    source: str
    modality: str
    domain: str
    title: str
    text: str
    image_ref: Optional[str] = None
    dense_vector: List[float] = []
    sparse_terms: List[str] = []


class RetrievalHit(BaseModel):
    unit_id: str
    source: str
    modality: str
    domain: str
    title: str
    text: str
    image_ref: Optional[str]
    dense_score: float
    sparse_score: float
    hybrid_score: float


class MemoryRecord(BaseModel):
    memory_id: str
    user_id: str
    summary: str
    tags: List[str] = []


class RetrievalEvaluation(BaseModel):
    context_precision: float
    context_recall: float
    needs_rewrite: bool
    reason: str


class AnswerEvaluation(BaseModel):
    response_relevancy: float
    faithfulness: float
    needs_human_review: bool
    reason: str


class RagasScoreCard(BaseModel):
    context_precision: float
    context_recall: float
    response_relevancy: float
    faithfulness: float


class DemoResponse(BaseModel):
    route: str
    answer: str
    retrieval_evaluation: RetrievalEvaluation
    answer_evaluation: AnswerEvaluation
    ragas_scores: RagasScoreCard
    hits: List[RetrievalHit]
    used_memory: List[MemoryRecord] = []
