from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

Modality = Literal["text", "image", "text_image", "text_table"]
QueryMode = Literal["text", "image", "text_image"]


class ParsedBlock(BaseModel):
    block_id: str
    source: str
    title_path: str
    text: str
    image_refs: List[str] = Field(default_factory=list)
    table_refs: List[str] = Field(default_factory=list)
    base64_refs: List[str] = Field(default_factory=list)


class KnowledgeChunk(BaseModel):
    chunk_id: str
    source: str
    domain: str
    modality: Modality
    title: str
    text: str
    image_ref: Optional[str] = None
    table_ref: Optional[str] = None
    image_caption: Optional[str] = None
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
    tags: List[str] = Field(default_factory=list)
    dense_vector: List[float] = Field(default_factory=list)


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
    query_mode: QueryMode
    answer: str
    rewritten_query: Optional[str] = None
    retrieval_check: RetrievalCheck
    answer_check: AnswerCheck
    ragas_metrics: RagasMetrics
    hits: List[SearchHit]
    memory_used: List[MemoryItem] = Field(default_factory=list)
    short_term_summary: str = ""
    cache_hit: bool = False
    human_review_pending: bool = False
    thread_id: Optional[str] = None
    workflow_steps: List[str] = Field(default_factory=list)
