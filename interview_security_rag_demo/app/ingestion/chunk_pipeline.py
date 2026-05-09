from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.models import DocumentChunk


@dataclass
class ChunkingPlan:
    strategy_name: str
    description: str
    output_examples: List[str]


class ChapterSemanticChunkPipeline:
    """
    Architecture module for:
    1. chapter / paragraph splitting
    2. semantic chunk refinement
    """

    def describe(self) -> ChunkingPlan:
        return ChunkingPlan(
            strategy_name="chapter_plus_semantic_chunking",
            description=(
                "Split by chapter headings first, then re-split long paragraphs with a semantic chunker to reduce cross-page "
                "table and interface-description fragmentation."
            ),
            output_examples=[
                "设备接入与离线排查 / 摄像头离线排查",
                "告警处置与误报治理 / 告警联动建议",
            ],
        )

    def refine(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        The runnable demo keeps original chunks.
        A full implementation would call a semantic chunker model here.
        """
        return chunks
