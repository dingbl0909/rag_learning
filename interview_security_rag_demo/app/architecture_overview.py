from __future__ import annotations

from typing import Any, Dict, List

from app.config import (
    EMBEDDING_MODEL_NAME,
    MILVUS_BM25_B,
    MILVUS_BM25_K1,
    MILVUS_DENSE_DIM,
    MILVUS_HNSW_EF_CONSTRUCTION,
    MILVUS_HNSW_M,
    SEMANTIC_CHUNKER_NAME,
    TXT_ALIGNMENT_ITEMS,
    UNSTRUCTURED_STRATEGY,
)


def build_architecture_overview() -> Dict[str, Any]:
    return {
        "project_positioning": "Interview-oriented security RAG demo with runnable core and architecture-complete modules.",
        "txt_alignment_items": list(TXT_ALIGNMENT_ITEMS),
        "runnable_core": [
            "FastAPI chat API",
            "Local hybrid retrieval demo",
            "Rule-based workflow routing",
            "Human review continuation flow",
        ],
        "extended_modules": [
            "Milvus cluster deployment blueprint",
            "PyMuPDF / Unstructured / Tesseract ingestion adapters",
            "BGE dense embedding wrapper",
            "LangGraph blueprint nodes and state",
            "Corrective RAG / Adaptive RAG evaluators",
            "RAGAS evaluation loop facade",
        ],
        "key_parameters": {
            "embedding_model": EMBEDDING_MODEL_NAME,
            "semantic_chunker": SEMANTIC_CHUNKER_NAME,
            "unstructured_strategy": UNSTRUCTURED_STRATEGY,
            "dense_dimension": MILVUS_DENSE_DIM,
            "bm25_k1": MILVUS_BM25_K1,
            "bm25_b": MILVUS_BM25_B,
            "hnsw_m": MILVUS_HNSW_M,
            "hnsw_ef_construction": MILVUS_HNSW_EF_CONSTRUCTION,
        },
    }
