from __future__ import annotations

from typing import List

from app.models import KnowledgeChunk, ParsedBlock


class ChapterSemanticChunkerDemo:
    """
    Demo for:
    chapter splitting + semantic chunker + image context binding.
    """

    def __init__(self, max_chars: int = 220):
        self.max_chars = max_chars

    def build_chunks(self, blocks: List[ParsedBlock]) -> List[KnowledgeChunk]:
        chunks: List[KnowledgeChunk] = []
        for block in blocks:
            pieces = self._split_long_text(block.text)
            domain = self._infer_domain(block)
            if block.media_refs:
                for media in block.media_refs:
                    for idx, piece in enumerate(pieces, start=1):
                        chunks.append(
                            KnowledgeChunk(
                                chunk_id=f"{block.block_id}-mm-{idx}-{media}",
                                source=block.source,
                                domain=domain,
                                modality="text_image",
                                title=block.title_path,
                                text=piece,
                                image_ref=media,
                            )
                        )
            else:
                for idx, piece in enumerate(pieces, start=1):
                    chunks.append(
                        KnowledgeChunk(
                            chunk_id=f"{block.block_id}-text-{idx}",
                            source=block.source,
                            domain=domain,
                            modality="text",
                            title=block.title_path,
                            text=piece,
                        )
                    )
        return chunks

    def _infer_domain(self, block: ParsedBlock) -> str:
        text = f"{block.source} {block.title_path}"
        if "告警" in text:
            return "alarm"
        if "部署" in text or "Milvus" in text:
            return "deployment"
        if "抓拍" in text or "snapshot" in text.lower():
            return "snapshot"
        return "device"

    def _split_long_text(self, text: str) -> List[str]:
        if len(text) <= self.max_chars:
            return [text]
        pieces: List[str] = []
        current = ""
        for sentence in text.replace("。", "。\n").splitlines():
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current) + len(sentence) <= self.max_chars:
                current += sentence
            else:
                if current:
                    pieces.append(current)
                current = sentence
        if current:
            pieces.append(current)
        return pieces
