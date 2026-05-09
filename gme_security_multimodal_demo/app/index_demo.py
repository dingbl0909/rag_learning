from __future__ import annotations

import math
from collections import Counter
from typing import List, Optional

from app.embedding_demo import GMEEmbeddingDemo, tokenize
from app.models import KnowledgeChunk, SearchHit


class HybridIndexDemo:
    """
    Dense + sparse hybrid retrieval demo:
    - Dense side: GME-like unified embedding
    - Sparse side: BM25-like keyword scoring
    - Fusion: weighted ranker
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
            if chunk.image_ref:
                chunk.dense_vector = self.encoder.encode_text_image(chunk.text, chunk.image_ref)
            else:
                chunk.dense_vector = self.encoder.encode_text(chunk.text)
            chunk.sparse_terms = tokenize(f"{chunk.title} {chunk.text}")
            total_len += len(chunk.sparse_terms)
            for term in set(chunk.sparse_terms):
                self.doc_freq[term] += 1
        self.avgdl = total_len / len(self.chunks) if self.chunks else 0.0

    def search(self, query: str, image_ref: Optional[str] = None, domain: Optional[str] = None, top_k: int = 4) -> List[SearchHit]:
        if image_ref:
            q_dense = self.encoder.encode_text_image(query, image_ref)
        else:
            q_dense = self.encoder.encode_text(query)
        q_terms = Counter(tokenize(query))

        hits: List[SearchHit] = []
        for chunk in self.chunks:
            if domain and chunk.domain != domain:
                continue
            dense_score = self._cosine(q_dense, chunk.dense_vector)
            sparse_score = self._bm25(q_terms, chunk)
            hybrid_score = dense_score * 0.65 + sparse_score * 0.35
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
