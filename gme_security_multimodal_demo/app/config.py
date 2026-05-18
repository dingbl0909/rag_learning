from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.embedding_demo import GMEEmbeddingModel


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class AppSettings:
    gme_model_path: str = "iic/gme-Qwen2-VL-7B-Instruct"
    gme_device: Optional[str] = None
    gme_image_root: Optional[Path] = None
    gme_trust_remote_code: bool = True
    gme_normalize_embeddings: bool = True
    use_milvus: bool = True
    use_redis: bool = True
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_token: Optional[str] = None
    milvus_kb_collection: str = "security_multimodal_kb"
    milvus_memory_collection: str = "security_long_term_memory"
    milvus_rebuild_index: bool = True
    redis_url: str = "redis://127.0.0.1:6379/0"
    redis_prefix: str = "gme_security_rag"

    @classmethod
    def from_env(cls) -> "AppSettings":
        image_root = os.getenv("GME_IMAGE_ROOT")
        return cls(
            gme_model_path=os.getenv("GME_MODEL_PATH", cls.gme_model_path),
            gme_device=os.getenv("GME_DEVICE") or None,
            gme_image_root=Path(image_root).expanduser() if image_root else None,
            gme_trust_remote_code=_env_bool("GME_TRUST_REMOTE_CODE", cls.gme_trust_remote_code),
            gme_normalize_embeddings=_env_bool("GME_NORMALIZE_EMBEDDINGS", cls.gme_normalize_embeddings),
            use_milvus=_env_bool("USE_MILVUS", cls.use_milvus),
            use_redis=_env_bool("USE_REDIS", cls.use_redis),
            milvus_uri=os.getenv("MILVUS_URI", cls.milvus_uri),
            milvus_token=os.getenv("MILVUS_TOKEN") or None,
            milvus_kb_collection=os.getenv("MILVUS_KB_COLLECTION", cls.milvus_kb_collection),
            milvus_memory_collection=os.getenv("MILVUS_MEMORY_COLLECTION", cls.milvus_memory_collection),
            milvus_rebuild_index=_env_bool("MILVUS_REBUILD_INDEX", cls.milvus_rebuild_index),
            redis_url=os.getenv("REDIS_URL", cls.redis_url),
            redis_prefix=os.getenv("REDIS_PREFIX", cls.redis_prefix),
        )


def build_embedding_encoder(settings: Optional[AppSettings] = None) -> GMEEmbeddingModel:
    settings = settings or AppSettings.from_env()
    return GMEEmbeddingModel(
        model_name_or_path=settings.gme_model_path,
        device=settings.gme_device,
        image_root=settings.gme_image_root,
        trust_remote_code=settings.gme_trust_remote_code,
        normalize_embeddings=settings.gme_normalize_embeddings,
    )
