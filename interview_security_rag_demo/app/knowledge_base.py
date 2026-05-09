from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from app.chunking import load_chunks_from_dir
from app.config import DOCS_DIR, TOP_K
from app.models import DocumentChunk, SearchHit
from app.retrieval import HybridRetriever


@dataclass
class KnowledgeBase:
    chunks: List[DocumentChunk]
    retriever: HybridRetriever

    @classmethod
    def build(cls) -> "KnowledgeBase":
        chunks = load_chunks_from_dir(DOCS_DIR)
        return cls(chunks=chunks, retriever=HybridRetriever(chunks))

    def search(self, query: str, domain: Optional[str] = None, top_k: int = TOP_K) -> List[SearchHit]:
        return self.retriever.search(query=query, domain=domain, top_k=top_k)
