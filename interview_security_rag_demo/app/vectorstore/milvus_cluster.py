from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class MilvusNode:
    name: str
    role: str
    replicas: int


@dataclass
class MilvusClusterBlueprint:
    deployment_mode: str
    nodes: List[MilvusNode]
    persistence_backend: str
    scaling_notes: List[str]


def build_cluster_blueprint() -> MilvusClusterBlueprint:
    return MilvusClusterBlueprint(
        deployment_mode="docker-compose for demo, kubernetes or distributed deployment for production",
        nodes=[
            MilvusNode(name="etcd", role="metadata", replicas=1),
            MilvusNode(name="minio", role="object-storage", replicas=1),
            MilvusNode(name="standalone-milvus", role="query-and-index", replicas=1),
        ],
        persistence_backend="local volume in demo, object storage in production",
        scaling_notes=[
            "Horizontal scaling would split query, data, and index roles.",
            "Interview demo keeps the distributed topology as a blueprint rather than a hard dependency.",
        ],
    )


def render_docker_compose_snippet() -> str:
    return """version: '3.8'
services:
  etcd:
    image: quay.io/coreos/etcd:v3.5.5
  minio:
    image: minio/minio:latest
  milvus:
    image: milvusdb/milvus:v2.4.0
    depends_on:
      - etcd
      - minio
"""
