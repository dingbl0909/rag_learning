from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from app.config import MILVUS_BM25_B, MILVUS_BM25_K1


@dataclass
class SparseEncoderConfig:
    algorithm: str = "BM25"
    posting_strategy: str = "DAAT"
    k1: float = MILVUS_BM25_K1
    b: float = MILVUS_BM25_B


class BM25SparseEncoder:
    """Blueprint sparse encoder matching the txt retrieval description."""

    def __init__(self, config: SparseEncoderConfig | None = None):
        self.config = config or SparseEncoderConfig()

    def describe(self) -> Dict[str, float | str]:
        return {
            "algorithm": self.config.algorithm,
            "posting_strategy": self.config.posting_strategy,
            "k1": self.config.k1,
            "b": self.config.b,
            "status": "blueprint",
        }
