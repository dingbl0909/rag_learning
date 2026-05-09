from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional

from app.models import DocumentChunk, SearchHit


def tokenize(text: str) -> List[str]:
    lowered = text.lower()
    tokens = re.findall(r"[a-z0-9_]+", lowered)
    chinese_blocks = re.findall(r"[\u4e00-\u9fff]+", text)
    for block in chinese_blocks:
        tokens.extend(list(block))
        if len(block) >= 2:
            tokens.extend(block[index : index + 2] for index in range(len(block) - 1))
    return [token for token in tokens if token.strip()]


class HybridRetriever:
    def __init__(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        self.term_frequencies: Dict[str, Counter[str]] = {}
        self.document_lengths: Dict[str, int] = {}
        self.document_frequency: Counter[str] = Counter()
        self.avgdl = 0.0
        self.dense_vectors: Dict[str, Dict[str, float]] = {}
        self._build_index()

    def _build_index(self) -> None:
        total_length = 0
        for chunk in self.chunks:
            tokens = tokenize(f"{chunk.title} {chunk.content}")
            tf = Counter(tokens)
            self.term_frequencies[chunk.chunk_id] = tf
            self.document_lengths[chunk.chunk_id] = len(tokens)
            total_length += len(tokens)
            for token in tf:
                self.document_frequency[token] += 1

        self.avgdl = total_length / len(self.chunks) if self.chunks else 0.0

        for chunk in self.chunks:
            tf = self.term_frequencies[chunk.chunk_id]
            dense_vector: Dict[str, float] = {}
            for token, count in tf.items():
                dense_vector[token] = (1.0 + math.log(count)) * self._idf(token)
            self.dense_vectors[chunk.chunk_id] = dense_vector

    def _idf(self, token: str) -> float:
        df = self.document_frequency.get(token, 0)
        n = len(self.chunks)
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def _cosine(self, left: Dict[str, float], right: Dict[str, float]) -> float:
        if not left or not right:
            return 0.0
        numerator = sum(left[token] * right.get(token, 0.0) for token in left)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return numerator / (left_norm * right_norm)

    def _build_query_vector(self, query: str) -> Dict[str, float]:
        tf = Counter(tokenize(query))
        return {token: (1.0 + math.log(count)) * self._idf(token) for token, count in tf.items()}

    def _bm25(self, query_tokens: List[str], chunk_id: str, k1: float = 1.5, b: float = 0.75) -> float:
        tf = self.term_frequencies[chunk_id]
        dl = self.document_lengths[chunk_id]
        score = 0.0
        for token in query_tokens:
            if token not in tf:
                continue
            idf = self._idf(token)
            freq = tf[token]
            denom = freq + k1 * (1 - b + b * dl / max(self.avgdl, 1.0))
            score += idf * (freq * (k1 + 1.0) / denom)
        return score

    def search(self, query: str, top_k: int, domain: Optional[str] = None) -> List[SearchHit]:
        query_tokens = tokenize(query)
        query_vector = self._build_query_vector(query)
        candidates = [chunk for chunk in self.chunks if domain is None or chunk.domain == domain]
        hits: List[SearchHit] = []

        for chunk in candidates:
            dense_score = self._cosine(query_vector, self.dense_vectors[chunk.chunk_id])
            sparse_score = self._bm25(query_tokens, chunk.chunk_id)
            hybrid_score = dense_score * 0.6 + sparse_score * 0.4
            hits.append(
                SearchHit(
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    domain=chunk.domain,
                    title=chunk.title,
                    content=chunk.content,
                    dense_score=round(dense_score, 4),
                    sparse_score=round(sparse_score, 4),
                    hybrid_score=round(hybrid_score, 4),
                )
            )

        hits.sort(key=lambda item: item.hybrid_score, reverse=True)
        return hits[:top_k]
