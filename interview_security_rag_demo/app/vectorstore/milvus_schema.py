from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

from app.config import (
    MILVUS_BM25_B,
    MILVUS_BM25_K1,
    MILVUS_COLLECTION_NAME,
    MILVUS_CONTEXT_COLLECTION_NAME,
    MILVUS_DENSE_DIM,
    MILVUS_HNSW_EF_CONSTRUCTION,
    MILVUS_HNSW_M,
)


@dataclass
class IndexSpec:
    field_name: str
    index_type: str
    metric_type: str
    params: Dict[str, float | int]


@dataclass
class CollectionBlueprint:
    collection_name: str
    primary_key: str
    vector_fields: List[str]
    scalar_fields: List[str]
    index_specs: List[IndexSpec]

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["index_specs"] = [asdict(spec) for spec in self.index_specs]
        return payload


def build_knowledge_collection_blueprint() -> CollectionBlueprint:
    return CollectionBlueprint(
        collection_name=MILVUS_COLLECTION_NAME,
        primary_key="chunk_id",
        vector_fields=["dense_vector", "sparse_vector"],
        scalar_fields=["source", "domain", "title", "content"],
        index_specs=[
            IndexSpec(
                field_name="dense_vector",
                index_type="HNSW",
                metric_type="IP",
                params={"M": MILVUS_HNSW_M, "efConstruction": MILVUS_HNSW_EF_CONSTRUCTION, "dim": MILVUS_DENSE_DIM},
            ),
            IndexSpec(
                field_name="sparse_vector",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
                params={"bm25_k1": MILVUS_BM25_K1, "bm25_b": MILVUS_BM25_B, "posting_algo": "DAAT"},
            ),
        ],
    )


def build_context_collection_blueprint() -> CollectionBlueprint:
    return CollectionBlueprint(
        collection_name=MILVUS_CONTEXT_COLLECTION_NAME,
        primary_key="context_id",
        vector_fields=["context_dense_vector", "context_sparse_vector"],
        scalar_fields=["user_id", "message_role", "context_text"],
        index_specs=[
            IndexSpec(field_name="context_dense_vector", index_type="HNSW", metric_type="IP", params={"M": 16, "efConstruction": 64}),
            IndexSpec(field_name="context_sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25", params={"posting_algo": "DAAT"}),
        ],
    )
