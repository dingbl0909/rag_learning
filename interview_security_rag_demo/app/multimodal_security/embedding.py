from __future__ import annotations

import math
import re
from typing import Iterable, List


def tokenize(text: str) -> List[str]:
    english = re.findall(r"[a-z0-9_]+", text.lower())
    chinese = re.findall(r"[\u4e00-\u9fff]+", text)
    tokens = list(english)
    for block in chinese:
        tokens.extend(list(block))
        if len(block) >= 2:
            tokens.extend(block[i : i + 2] for i in range(len(block) - 1))
    return [token for token in tokens if token]


class GMEMultimodalEncoder:
    """
    Interview-friendly mock for iic/gme-Qwen2-VL-7B-Instruct.

    It keeps the mental model of a unified vector space for text, image,
    and text-image pairs, without requiring a real model service.
    """

    def __init__(self, dim: int = 16):
        self.dim = dim

    def encode_text(self, text: str) -> List[float]:
        return self._hash_tokens(tokenize(text))

    def encode_image(self, image_ref: str) -> List[float]:
        return self._hash_tokens([image_ref, "image", "vision"])

    def encode_text_image(self, text: str, image_ref: str) -> List[float]:
        return self._merge(self.encode_text(text), self.encode_image(image_ref))

    def _hash_tokens(self, tokens: Iterable[str]) -> List[float]:
        vector = [0.0] * self.dim
        for token in tokens:
            bucket = sum(ord(char) for char in token) % self.dim
            vector[bucket] += 1.0 + (len(token) % 3) * 0.2
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [round(value / norm, 6) for value in vector]

    def _merge(self, left: List[float], right: List[float]) -> List[float]:
        merged = [(l + r) / 2.0 for l, r in zip(left, right)]
        norm = math.sqrt(sum(value * value for value in merged)) or 1.0
        return [round(value / norm, 6) for value in merged]
