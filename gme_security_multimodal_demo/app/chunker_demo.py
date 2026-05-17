from __future__ import annotations

from typing import List, Set

from app.embedding_demo import tokenize
from app.models import KnowledgeChunk, ParsedBlock


class ChapterSemanticChunkerDemo:
    """
    Chapter splitting + semantic chunker + image caption binding.

    Produces text / image / text_image / text_table knowledge units.
    """

    def __init__(self, max_chars: int = 220, semantic_threshold: float = 0.35):
        self.max_chars = max_chars
        self.semantic_threshold = semantic_threshold

    def build_chunks(self, blocks: List[ParsedBlock]) -> List[KnowledgeChunk]:
        chunks: List[KnowledgeChunk] = []
        for block in blocks:
            domain = self._infer_domain(block)
            pieces = self._semantic_split(block.text) if block.text else []
            if not pieces and (block.image_refs or block.table_refs):
                pieces = [""]

            for image_ref in block.image_refs:
                caption = self._build_image_caption(block, image_ref)
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=f"{block.block_id}-img-{image_ref}",
                        source=block.source,
                        domain=domain,
                        modality="image",
                        title=block.title_path,
                        text=caption,
                        image_ref=image_ref,
                        image_caption=caption,
                    )
                )
                for idx, piece in enumerate(pieces, start=1):
                    merged = f"{piece} {caption}".strip() if piece else caption
                    chunks.append(
                        KnowledgeChunk(
                            chunk_id=f"{block.block_id}-mm-{idx}-{image_ref}",
                            source=block.source,
                            domain=domain,
                            modality="text_image",
                            title=block.title_path,
                            text=merged,
                            image_ref=image_ref,
                            image_caption=caption,
                        )
                    )

            for table_ref in block.table_refs:
                table_text = block.text or f"表格 {table_ref} 关联章节：{block.title_path}"
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=f"{block.block_id}-table-{table_ref}",
                        source=block.source,
                        domain=domain,
                        modality="text_table",
                        title=block.title_path,
                        text=table_text,
                        table_ref=table_ref,
                    )
                )

            if block.text and not block.image_refs:
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

    def _build_image_caption(self, block: ParsedBlock, image_ref: str) -> str:
        context = block.text.strip() or block.title_path
        base64_note = f"，OCR载荷={block.base64_refs[0]}" if block.base64_refs else ""
        return f"[图语义] {image_ref}：{context[:120]}{base64_note}"

    def _infer_domain(self, block: ParsedBlock) -> str:
        text = f"{block.source} {block.title_path} {block.text}"
        if "告警" in text:
            return "alarm"
        if "部署" in text or "Milvus" in text:
            return "deployment"
        if "抓拍" in text or "snapshot" in text.lower():
            return "snapshot"
        return "device"

    def _semantic_split(self, text: str) -> List[str]:
        if len(text) <= self.max_chars:
            return [text] if text else []
        sentences = [s.strip() for s in text.replace("。", "。\n").splitlines() if s.strip()]
        if not sentences:
            return [text]
        groups: List[List[str]] = [[sentences[0]]]
        for sentence in sentences[1:]:
            prev = groups[-1][-1]
            if self._sentence_similarity(prev, sentence) >= self.semantic_threshold:
                groups[-1].append(sentence)
            else:
                groups.append([sentence])
        pieces: List[str] = []
        for group in groups:
            merged = "".join(group)
            if len(merged) <= self.max_chars:
                pieces.append(merged)
            else:
                pieces.extend(self._split_by_length(merged))
        return pieces

    def _sentence_similarity(self, left: str, right: str) -> float:
        a: Set[str] = set(tokenize(left))
        b: Set[str] = set(tokenize(right))
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def _split_by_length(self, text: str) -> List[str]:
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
