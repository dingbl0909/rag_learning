from __future__ import annotations

import math
from collections import Counter
from typing import List, Optional

from app.multimodal_security.embedding import GMEMultimodalEncoder, tokenize
from app.multimodal_security.models import KnowledgeUnit, RetrievalHit


class HybridMultimodalIndex:
    """
    Dense + sparse demo index mirroring:
    GME dense embedding + BM25 sparse retrieval + weighted fusion.
    """

    def __init__(self, units: List[KnowledgeUnit], encoder: GMEMultimodalEncoder):
        self.units = units
        self.encoder = encoder
        self.doc_freq = Counter()
        self.avgdl = 0.0
        self._prepare_units()

    def _prepare_units(self) -> None:
        total_len = 0
        for unit in self.units:
            if unit.image_ref:
                unit.dense_vector = self.encoder.encode_text_image(unit.text, unit.image_ref)
            else:
                unit.dense_vector = self.encoder.encode_text(unit.text)
            unit.sparse_terms = tokenize(f"{unit.title} {unit.text}")
            total_len += len(unit.sparse_terms)
            for term in set(unit.sparse_terms):
                self.doc_freq[term] += 1
        self.avgdl = total_len / len(self.units) if self.units else 0.0

    def search(self, query: str, image_ref: Optional[str] = None, domain: Optional[str] = None, top_k: int = 4) -> List[RetrievalHit]:
        query_dense = self.encoder.encode_text_image(query, image_ref) if image_ref else self.encoder.encode_text(query)
        query_terms = tokenize(query)
        query_term_counter = Counter(query_terms)
        hits: List[RetrievalHit] = []

        for unit in self.units:
            if domain and unit.domain != domain:
                continue
            dense_score = self._cosine(query_dense, unit.dense_vector)
            sparse_score = self._bm25(query_term_counter, unit)
            hybrid_score = dense_score * 0.65 + sparse_score * 0.35
            hits.append(
                RetrievalHit(
                    unit_id=unit.unit_id,
                    source=unit.source,
                    modality=unit.modality,
                    domain=unit.domain,
                    title=unit.title,
                    text=unit.text,
                    image_ref=unit.image_ref,
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
        total_docs = len(self.units)
        df = self.doc_freq.get(term, 0)
        return math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)

    def _bm25(self, query_terms: Counter[str], unit: KnowledgeUnit, k1: float = 1.5, b: float = 0.75) -> float:
        doc_terms = Counter(unit.sparse_terms)
        dl = len(unit.sparse_terms)
        score = 0.0
        for term in query_terms:
            if term not in doc_terms:
                continue
            tf = doc_terms[term]
            idf = self._idf(term)
            denom = tf + k1 * (1 - b + b * dl / max(self.avgdl, 1.0))
            score += idf * (tf * (k1 + 1.0) / denom)
        return score
