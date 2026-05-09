from __future__ import annotations

import math
import re
from typing import Iterable, List


def tokenize(text: str) -> List[str]:
    english = re.findall(r"[a-z0-9_]+", text.lower())
    chinese_blocks = re.findall(r"[\u4e00-\u9fff]+", text)
    tokens = list(english)
    for block in chinese_blocks:
        tokens.extend(list(block))
        if len(block) >= 2:
            tokens.extend(block[i : i + 2] for i in range(len(block) - 1))
    return [token for token in tokens if token]


class GMEEmbeddingDemo:
    """
    A readable mock of iic/gme-Qwen2-VL-7B-Instruct.

    It keeps the concept of a unified vector space for text, image, and
    text-image pairs without depending on an actual model service.
    """

    def __init__(self, dim: int = 16):
        self.dim = dim

    def encode_text(self, text: str) -> List[float]:
        return self._normalize(self._bucketize(tokenize(text)))

    def encode_image(self, image_ref: str) -> List[float]:
        return self._normalize(self._bucketize([image_ref, "image", "vision"]))

    def encode_text_image(self, text: str, image_ref: str) -> List[float]:
        left = self.encode_text(text)
        right = self.encode_image(image_ref)
        merged = [(l + r) / 2.0 for l, r in zip(left, right)]
        return self._normalize(merged)

    def _bucketize(self, tokens: Iterable[str]) -> List[float]:
        vector = [0.0] * self.dim
        for token in tokens:
            bucket = sum(ord(char) for char in token) % self.dim
            vector[bucket] += 1.0 + (len(token) % 3) * 0.25
        return vector

    def _normalize(self, vector: List[float]) -> List[float]:
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [round(value / norm, 6) for value in vector]
