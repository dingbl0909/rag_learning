from __future__ import annotations

import math
from collections import Counter
from typing import List, Optional

from app.embedding_demo import GMEEmbeddingDemo, tokenize
from app.models import KnowledgeChunk, QueryMode, SearchHit


class HybridIndexDemo:
    """
    Dense + sparse hybrid retrieval (Milvus dual-index concept).

    - Dense: GME-like unified embedding (HNSW concept)
    - Sparse: BM25-like scoring (DAAT / inverted index concept)
    - Supports text / image / text_image query modes (Any-to-Any)
    """

    def __init__(self, chunks: List[KnowledgeChunk], encoder: GMEEmbeddingDemo):
        self.chunks = chunks
        self.encoder = encoder
        self.doc_freq = Counter()
        self.avgdl = 0.0
        self._prepare()

    def _prepare(self) -> None:
        total_len = 0
        for chunk in self.chunks:
            chunk.dense_vector = self._encode_chunk(chunk)
            sparse_source = f"{chunk.title} {chunk.text} {chunk.image_caption or ''} {chunk.table_ref or ''}"
            chunk.sparse_terms = tokenize(sparse_source)
            total_len += len(chunk.sparse_terms)
            for term in set(chunk.sparse_terms):
                self.doc_freq[term] += 1
        self.avgdl = total_len / len(self.chunks) if self.chunks else 0.0

    def _encode_chunk(self, chunk: KnowledgeChunk) -> List[float]:
        if chunk.modality == "image" and chunk.image_ref:
            return self.encoder.encode_image(chunk.image_ref)
        if chunk.modality == "text_image" and chunk.image_ref:
            return self.encoder.encode_text_image(chunk.text, chunk.image_ref)
        return self.encoder.encode_text(chunk.text)

    def search(
        self,
        query: str = "",
        image_ref: Optional[str] = None,
        query_mode: Optional[QueryMode] = None,
        domain: Optional[str] = None,
        top_k: int = 4,
    ) -> List[SearchHit]:
        mode = query_mode or self._infer_query_mode(query, image_ref)
        q_dense, q_terms = self._encode_query(query, image_ref, mode)

        hits: List[SearchHit] = []
        for chunk in self.chunks:
            dense_score = self._cosine(q_dense, chunk.dense_vector)
            sparse_score = self._bm25(q_terms, chunk) if q_terms else 0.0
            hybrid_score = dense_score * 0.65 + sparse_score * 0.35
            if domain and chunk.domain == domain:
                hybrid_score += 0.05
            hits.append(
                SearchHit(
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    domain=chunk.domain,
                    modality=chunk.modality,
                    title=chunk.title,
                    text=chunk.text,
                    image_ref=chunk.image_ref,
                    dense_score=round(dense_score, 4),
                    sparse_score=round(sparse_score, 4),
                    hybrid_score=round(hybrid_score, 4),
                )
            )
        hits.sort(key=lambda item: item.hybrid_score, reverse=True)
        return hits[:top_k]

    def _infer_query_mode(self, query: str, image_ref: Optional[str]) -> QueryMode:
        if image_ref and not query.strip():
            return "image"
        if image_ref:
            return "text_image"
        return "text"

    def _encode_query(self, query: str, image_ref: Optional[str], mode: QueryMode):
        if mode == "image":
            assert image_ref
            return self.encoder.encode_image(image_ref), Counter()
        if mode == "text_image":
            assert image_ref
            return self.encoder.encode_text_image(query, image_ref), Counter(tokenize(query))
        return self.encoder.encode_text(query), Counter(tokenize(query))

    def _cosine(self, left: List[float], right: List[float]) -> float:
        return sum(l * r for l, r in zip(left, right))

    def _idf(self, term: str) -> float:
        total = len(self.chunks)
        df = self.doc_freq.get(term, 0)
        return math.log((total - df + 0.5) / (df + 0.5) + 1.0)

    def _bm25(self, query_terms: Counter[str], chunk: KnowledgeChunk, k1: float = 1.5, b: float = 0.75) -> float:
        terms = Counter(chunk.sparse_terms)
        dl = len(chunk.sparse_terms)
        score = 0.0
        for term in query_terms:
            if term not in terms:
                continue
            tf = terms[term]
            idf = self._idf(term)
            denom = tf + k1 * (1 - b + b * dl / max(self.avgdl, 1.0))
            score += idf * (tf * (k1 + 1.0) / denom)
        return score
