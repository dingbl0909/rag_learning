from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "data" / "security_docs"
DEPLOY_DIR = BASE_DIR / "deploy"
ARCH_DOCS_DIR = BASE_DIR / "docs"

MAX_CHUNK_CHARS = 260
TOP_K = 3
LOW_CONFIDENCE_THRESHOLD = 0.42
RISK_KEYWORDS = ("重启", "升级", "删除", "停机", "变更", "关闭告警")

# Extended architecture settings aligned with the txt design.
MILVUS_COLLECTION_NAME = "security_knowledge_chunks"
MILVUS_CONTEXT_COLLECTION_NAME = "security_dialog_context"
MILVUS_DENSE_DIM = 1024
MILVUS_BM25_K1 = 1.5
MILVUS_BM25_B = 0.75
MILVUS_HNSW_M = 30
MILVUS_HNSW_EF_CONSTRUCTION = 64

EMBEDDING_MODEL_NAME = "BAAI/bge-large-zh-v1.5"
SEMANTIC_CHUNKER_NAME = "SemanticChunker"
UNSTRUCTURED_STRATEGY = "hi_res"

TXT_ALIGNMENT_ITEMS = (
    "Milvus cluster and Docker deployment",
    "PyMuPDF + Unstructured + Tesseract ingestion",
    "Chapter chunking + semantic chunking",
    "BGE dense embedding + BM25 sparse retrieval",
    "DAAT / BM25 / HNSW index tuning",
    "LangGraph dynamic routing workflow",
    "Corrective RAG and Adaptive RAG evaluation",
    "RAGAS assessment loop",
)
