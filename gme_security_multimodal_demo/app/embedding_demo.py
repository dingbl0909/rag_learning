from __future__ import annotations

import base64
import binascii
import inspect
import io
import re
from pathlib import Path
from typing import Any, List, Optional, Protocol


def tokenize(text: str) -> List[str]:
    english = re.findall(r"[a-z0-9_]+", text.lower())
    chinese_blocks = re.findall(r"[\u4e00-\u9fff]+", text)
    tokens = list(english)
    for block in chinese_blocks:
        tokens.extend(list(block))
        if len(block) >= 2:
            tokens.extend(block[i : i + 2] for i in range(len(block) - 1))
    return [token for token in tokens if token]


class EmbeddingEncoder(Protocol):
    def encode_text(self, text: str) -> List[float]:
        ...

    def encode_image(self, image_ref: str) -> List[float]:
        ...

    def encode_text_image(self, text: str, image_ref: str) -> List[float]:
        ...


class GMEEmbeddingModel:
    """
    SentenceTransformers adapter for iic/gme-Qwen2-VL-7B-Instruct.

    image_ref supports:
    - absolute / relative image file path
    - filename under image_root
    - raw base64 string
    - data:image/...;base64,... URL
    """

    IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def __init__(
        self,
        model_name_or_path: str = "iic/gme-Qwen2-VL-7B-Instruct",
        device: Optional[str] = None,
        image_root: Optional[Path] = None,
        trust_remote_code: bool = True,
        normalize_embeddings: bool = True,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.image_root = image_root
        self.trust_remote_code = trust_remote_code
        self.normalize_embeddings = normalize_embeddings
        self._model: Any = None

    def encode_text(self, text: str) -> List[float]:
        if not text.strip():
            raise ValueError("encode_text requires non-empty text")
        return self._encode_payload(text)

    def encode_image(self, image_ref: str) -> List[float]:
        image = self._load_image(image_ref)
        return self._encode_image_payload(image)

    def encode_text_image(self, text: str, image_ref: str) -> List[float]:
        if not text.strip():
            return self.encode_image(image_ref)
        image = self._load_image(image_ref)
        return self._encode_text_image_payload(text, image)

    @property
    def model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for the production GME encoder. "
                    "Install requirements.txt in the deployment environment."
                ) from exc

            kwargs: dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
            if self.device:
                kwargs["device"] = self.device
            self._model = SentenceTransformer(self.model_name_or_path, **kwargs)
        return self._model

    def _encode_payload(self, payload: Any) -> List[float]:
        return self._first_vector(self._call_encode([payload]))

    def _encode_image_payload(self, image: Any) -> List[float]:
        errors: List[str] = []
        for args, kwargs in (
            ((), {"images": [image]}),
            (([image],), {}),
            (([{"image": image}],), {}),
        ):
            try:
                return self._first_vector(self._call_encode(*args, **kwargs))
            except TypeError as exc:
                errors.append(str(exc))
        raise RuntimeError("GME model does not expose a compatible image encode interface: " + " | ".join(errors))

    def _encode_text_image_payload(self, text: str, image: Any) -> List[float]:
        errors: List[str] = []
        for args, kwargs in (
            (([{"text": text, "image": image}],), {}),
            ((), {"sentences": [text], "images": [image]}),
            (([text],), {"images": [image]}),
        ):
            try:
                return self._first_vector(self._call_encode(*args, **kwargs))
            except TypeError as exc:
                errors.append(str(exc))
        raise RuntimeError("GME model does not expose a compatible text-image encode interface: " + " | ".join(errors))

    def _call_encode(self, *args: Any, **kwargs: Any) -> Any:
        kwargs = dict(kwargs)
        if self._supports_encode_kwarg("normalize_embeddings"):
            kwargs.setdefault("normalize_embeddings", self.normalize_embeddings)
        try:
            return self.model.encode(*args, convert_to_numpy=True, show_progress_bar=False, **kwargs)
        except TypeError:
            return self.model.encode(*args, **kwargs)

    def _supports_encode_kwarg(self, name: str) -> bool:
        try:
            signature = inspect.signature(self.model.encode)
        except (TypeError, ValueError):
            return True
        return name in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    def _load_image(self, image_ref: str) -> Any:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required to load image inputs for GME.") from exc

        path = self._resolve_image_path(image_ref)
        if path:
            return Image.open(path).convert("RGB")

        payload = image_ref.split(",", 1)[1] if image_ref.startswith("data:image/") and "," in image_ref else image_ref
        try:
            raw = base64.b64decode(payload, validate=True)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except (binascii.Error, ValueError, OSError) as exc:
            raise FileNotFoundError(
                f"Cannot resolve image_ref '{image_ref}'. Use an image path, a file under GME_IMAGE_ROOT, "
                "or a base64 image payload."
            ) from exc

    def _resolve_image_path(self, image_ref: str) -> Optional[Path]:
        candidates = [Path(image_ref).expanduser()]
        if self.image_root:
            candidates.append(self.image_root / image_ref)
            if not Path(image_ref).suffix:
                candidates.extend(self.image_root / f"{image_ref}{suffix}" for suffix in self.IMAGE_SUFFIXES)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    def _first_vector(self, encoded: Any) -> List[float]:
        vector = encoded[0] if hasattr(encoded, "__len__") and len(encoded) and not isinstance(encoded[0], (float, int)) else encoded
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        return [float(value) for value in vector]


GMEEmbeddingDemo = GMEEmbeddingModel
