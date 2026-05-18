from __future__ import annotations

import hashlib
import json
import time
from types import SimpleNamespace
from typing import List, Optional, Tuple

from app.config import AppSettings
from app.embedding_demo import EmbeddingEncoder
from app.models import MemoryItem, SearchHit


class RedisMilvusMemoryEngine:
    """
    Production-like memory layer.

    - Redis: short-term turns and query result cache
    - Milvus: long-term semantic memory vectors
    """

    def __init__(self, encoder: EmbeddingEncoder, settings: AppSettings):
        self.encoder = encoder
        self.settings = settings
        self.redis = self._build_redis()
        self.milvus = self._build_milvus()
        self.collection_name = settings.milvus_memory_collection
        self.alias = "security_memory"

    def add_user_turn(self, user_id: str, message: str) -> None:
        key = self._key("short", user_id)
        self.redis.rpush(key, message)
        self.redis.ltrim(key, -6, -1)

    def summarize_short_term(self, user_id: str) -> str:
        history = [item.decode("utf-8") if isinstance(item, bytes) else item for item in self.redis.lrange(self._key("short", user_id), -3, -1)]
        if not history:
            return ""
        return "近期会话：" + " | ".join(history)

    def cache_key(self, user_id: str, question: str, image_ref: Optional[str]) -> str:
        raw = f"{user_id}|{question.strip()}|{image_ref or ''}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get_cached_hits(self, user_id: str, question: str, image_ref: Optional[str]) -> Optional[Tuple[List[SearchHit], str]]:
        raw = self.redis.get(self._key("cache", self.cache_key(user_id, question, image_ref)))
        if not raw:
            return None
        payload = json.loads(raw)
        hits = [SearchHit(**item) for item in payload["hits"]]
        return hits, payload["route"]

    def put_cached_hits(self, user_id: str, question: str, image_ref: Optional[str], hits: List[SearchHit], route: str) -> None:
        payload = {
            "route": route,
            "hits": [hit.model_dump() for hit in hits],
        }
        self.redis.setex(
            self._key("cache", self.cache_key(user_id, question, image_ref)),
            3600,
            json.dumps(payload, ensure_ascii=False),
        )

    def save_long_term(self, user_id: str, summary: str, tags: List[str]) -> MemoryItem:
        vector = self.encoder.encode_text(summary)
        self._ensure_memory_collection(len(vector))
        memory_id = f"{user_id}-{int(time.time() * 1000)}"
        item = MemoryItem(
            memory_id=memory_id,
            user_id=user_id,
            summary=summary,
            tags=tags,
            dense_vector=vector,
        )
        collection = self._memory_collection()
        collection.insert(
            [
                [item.memory_id],
                [item.user_id],
                [self._truncate(item.summary, 8192)],
                [json.dumps(item.tags, ensure_ascii=False)],
                [int(time.time())],
                [item.dense_vector],
            ]
        )
        collection.flush()
        return item

    def search_long_term(self, user_id: str, query: str = "", limit: int = 2) -> List[MemoryItem]:
        if not self.milvus.utility.has_collection(self.collection_name, using=self.alias):
            return []
        collection = self._memory_collection()
        collection.load()
        expr = f'user_id == "{self._escape_expr(user_id)}"'
        if not query.strip():
            rows = collection.query(expr=expr, output_fields=["memory_id", "user_id", "summary", "tags_json", "dense_vector"], limit=limit)
            return [self._memory_from_row(row) for row in rows]
        q_vec = self.encoder.encode_text(query)
        results = collection.search(
            data=[q_vec],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=limit,
            expr=expr,
            output_fields=["memory_id", "user_id", "summary", "tags_json", "dense_vector"],
        )
        return [self._memory_from_entity(hit.entity) for hit in results[0]]

    def _build_redis(self):
        import redis

        client = redis.Redis.from_url(self.settings.redis_url)
        client.ping()
        return client

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

    def _ensure_memory_collection(self, dim: int) -> None:
        if self.milvus.utility.has_collection(self.collection_name, using=self.alias):
            return
        fields = [
            self.milvus.FieldSchema(name="memory_id", dtype=self.milvus.DataType.VARCHAR, is_primary=True, max_length=256),
            self.milvus.FieldSchema(name="user_id", dtype=self.milvus.DataType.VARCHAR, max_length=128),
            self.milvus.FieldSchema(name="summary", dtype=self.milvus.DataType.VARCHAR, max_length=8192),
            self.milvus.FieldSchema(name="tags_json", dtype=self.milvus.DataType.VARCHAR, max_length=2048),
            self.milvus.FieldSchema(name="created_at", dtype=self.milvus.DataType.INT64),
            self.milvus.FieldSchema(name="dense_vector", dtype=self.milvus.DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = self.milvus.CollectionSchema(fields=fields, description="Security RAG long-term semantic memory")
        collection = self.milvus.Collection(name=self.collection_name, schema=schema, using=self.alias)
        collection.create_index(
            field_name="dense_vector",
            index_params={"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 30, "efConstruction": 64}},
        )
        collection.load()

    def _memory_collection(self):
        return self.milvus.Collection(name=self.collection_name, using=self.alias)

    def _key(self, kind: str, value: str) -> str:
        return f"{self.settings.redis_prefix}:{kind}:{value}"

    def _memory_from_row(self, row: dict) -> MemoryItem:
        return MemoryItem(
            memory_id=row["memory_id"],
            user_id=row["user_id"],
            summary=row["summary"],
            tags=json.loads(row.get("tags_json") or "[]"),
            dense_vector=row.get("dense_vector") or [],
        )

    def _memory_from_entity(self, entity) -> MemoryItem:
        return MemoryItem(
            memory_id=entity.get("memory_id"),
            user_id=entity.get("user_id"),
            summary=entity.get("summary"),
            tags=json.loads(entity.get("tags_json") or "[]"),
            dense_vector=entity.get("dense_vector") or [],
        )

    def _escape_expr(self, value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def _truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[: max_length - 3] + "..."
