from __future__ import annotations

import math
from collections import Counter
from types import SimpleNamespace
from typing import List, Optional

from app.config import AppSettings
from app.embedding_demo import EmbeddingEncoder, tokenize
from app.models import KnowledgeChunk, QueryMode, SearchHit


class MilvusHybridIndex:
    """
    Minimal production index.

    Milvus stores dense GME vectors and chunk metadata. Sparse BM25 is kept as a
    lightweight local reranker so the retrieval contract remains hybrid.
    """

    def __init__(self, chunks: List[KnowledgeChunk], encoder: EmbeddingEncoder, settings: AppSettings):
        self.chunks = chunks
        self.encoder = encoder
        self.settings = settings
        self.collection_name = settings.milvus_kb_collection
        self.alias = "security_kb"
        self.doc_freq: Counter[str] = Counter()
        self.avgdl = 0.0
        self.chunk_by_id: dict[str, KnowledgeChunk] = {}
        self.milvus = self._build_milvus()
        self._prepare()
        self._sync_collection()

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
        collection = self._collection()
        collection.load()
        results = collection.search(
            data=[q_dense],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=max(top_k * 8, top_k),
            output_fields=["chunk_id", "source", "domain", "modality", "title", "text", "image_ref"],
        )

        hits: List[SearchHit] = []
        for raw_hit in results[0]:
            entity = raw_hit.entity
            chunk_id = entity.get("chunk_id")
            chunk = self.chunk_by_id.get(chunk_id) or self._chunk_from_entity(entity)
            dense_score = float(getattr(raw_hit, "score", getattr(raw_hit, "distance", 0.0)))
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

    def _prepare(self) -> None:
        total_len = 0
        for chunk in self.chunks:
            chunk.dense_vector = self._encode_chunk(chunk)
            sparse_source = f"{chunk.title} {chunk.text} {chunk.image_caption or ''} {chunk.table_ref or ''}"
            chunk.sparse_terms = tokenize(sparse_source)
            self.chunk_by_id[chunk.chunk_id] = chunk
            total_len += len(chunk.sparse_terms)
            for term in set(chunk.sparse_terms):
                self.doc_freq[term] += 1
        self.avgdl = total_len / len(self.chunks) if self.chunks else 0.0

    def _sync_collection(self) -> None:
        if not self.chunks:
            raise ValueError("Cannot build Milvus index without knowledge chunks")
        if self.settings.milvus_rebuild_index and self.milvus.utility.has_collection(self.collection_name, using=self.alias):
            self.milvus.utility.drop_collection(self.collection_name, using=self.alias)
        if not self.milvus.utility.has_collection(self.collection_name, using=self.alias):
            self._create_collection(len(self.chunks[0].dense_vector))
            self._insert_chunks()
        self._collection().load()

    def _create_collection(self, dim: int) -> None:
        fields = [
            self.milvus.FieldSchema(name="chunk_id", dtype=self.milvus.DataType.VARCHAR, is_primary=True, max_length=256),
            self.milvus.FieldSchema(name="source", dtype=self.milvus.DataType.VARCHAR, max_length=512),
            self.milvus.FieldSchema(name="domain", dtype=self.milvus.DataType.VARCHAR, max_length=64),
            self.milvus.FieldSchema(name="modality", dtype=self.milvus.DataType.VARCHAR, max_length=64),
            self.milvus.FieldSchema(name="title", dtype=self.milvus.DataType.VARCHAR, max_length=1024),
            self.milvus.FieldSchema(name="text", dtype=self.milvus.DataType.VARCHAR, max_length=8192),
            self.milvus.FieldSchema(name="image_ref", dtype=self.milvus.DataType.VARCHAR, max_length=1024),
            self.milvus.FieldSchema(name="dense_vector", dtype=self.milvus.DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = self.milvus.CollectionSchema(fields=fields, description="Security multimodal RAG knowledge chunks")
        collection = self.milvus.Collection(name=self.collection_name, schema=schema, using=self.alias)
        collection.create_index(
            field_name="dense_vector",
            index_params={"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 30, "efConstruction": 64}},
        )

    def _insert_chunks(self) -> None:
        collection = self._collection()
        collection.insert(
            [
                [chunk.chunk_id for chunk in self.chunks],
                [chunk.source for chunk in self.chunks],
                [chunk.domain for chunk in self.chunks],
                [chunk.modality for chunk in self.chunks],
                [self._truncate(chunk.title, 1024) for chunk in self.chunks],
                [self._truncate(chunk.text, 8192) for chunk in self.chunks],
                [self._truncate(chunk.image_ref or "", 1024) for chunk in self.chunks],
                [chunk.dense_vector for chunk in self.chunks],
            ]
        )
        collection.flush()

    def _build_milvus(self):
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

        kwargs = {"alias": self.alias, "uri": self.settings.milvus_uri}
        if self.settings.milvus_token:
            kwargs["token"] = self.settings.milvus_token
        connections.connect(**kwargs)
        return SimpleNamespace(
            Collection=Collection,
            CollectionSchema=CollectionSchema,
            DataType=DataType,
            FieldSchema=FieldSchema,
            utility=utility,
        )

    def _collection(self):
        return self.milvus.Collection(name=self.collection_name, using=self.alias)

    def _encode_chunk(self, chunk: KnowledgeChunk) -> List[float]:
        if chunk.modality == "image" and chunk.image_ref:
            return self.encoder.encode_image(chunk.image_ref)
        if chunk.modality == "text_image" and chunk.image_ref:
            return self.encoder.encode_text_image(chunk.text, chunk.image_ref)
        return self.encoder.encode_text(chunk.text)

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

    def _chunk_from_entity(self, entity) -> KnowledgeChunk:
        chunk = KnowledgeChunk(
            chunk_id=entity.get("chunk_id"),
            source=entity.get("source"),
            domain=entity.get("domain"),
            modality=entity.get("modality"),
            title=entity.get("title"),
            text=entity.get("text"),
            image_ref=entity.get("image_ref") or None,
        )
        chunk.sparse_terms = tokenize(f"{chunk.title} {chunk.text}")
        return chunk

    def _truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[: max_length - 3] + "..."
