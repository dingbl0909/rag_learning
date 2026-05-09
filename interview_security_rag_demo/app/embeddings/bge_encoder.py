from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from app.config import EMBEDDING_MODEL_NAME, MILVUS_DENSE_DIM


@dataclass
class EmbeddingConfig:
    model_name: str = EMBEDDING_MODEL_NAME
    vector_dimension: int = MILVUS_DENSE_DIM
    deployment_mode: str = "private"


class BGELargeZhEncoder:
    """
    Blueprint for a private deployment of bge-large-zh-v1.5.

    The current runnable demo does not call a real embedding service.
    This module exists so the architecture matches the txt description.
    """

    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig()

    def encode_text(self, text: str) -> List[float]:
        seed = sum(ord(char) for char in text) % 97
        base = (seed + 1) / 100.0
        return [base] * min(self.config.vector_dimension, 16)

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.encode_text(text) for text in texts]

    def describe(self) -> Dict[str, str]:
        return {
            "module": "BGELargeZhEncoder",
            "model_name": self.config.model_name,
            "deployment_mode": self.config.deployment_mode,
            "status": "blueprint with mock output",
        }
