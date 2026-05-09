from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

from app.config import MAX_CHUNK_CHARS
from app.models import DocumentChunk


def infer_domain(file_path: Path) -> str:
    name = file_path.stem.lower()
    if "alarm" in name:
        return "alarm"
    if "deploy" in name:
        return "deployment"
    return "device"


def split_long_text(text: str, limit: int = MAX_CHUNK_CHARS) -> List[str]:
    text = text.strip()
    if len(text) <= limit:
        return [text]

    pieces: List[str] = []
    sentence_buffer = ""
    sentences = [item.strip() for item in re.split(r"(?<=[。！？；;.!?])", text) if item.strip()]
    for sentence in sentences:
        if len(sentence_buffer) + len(sentence) <= limit:
            sentence_buffer += sentence
            continue
        if sentence_buffer:
            pieces.append(sentence_buffer.strip())
        if len(sentence) <= limit:
            sentence_buffer = sentence
        else:
            for start in range(0, len(sentence), limit):
                pieces.append(sentence[start : start + limit].strip())
            sentence_buffer = ""
    if sentence_buffer:
        pieces.append(sentence_buffer.strip())
    return [piece for piece in pieces if piece]


def markdown_to_chunks(file_path: Path) -> List[DocumentChunk]:
    lines = file_path.read_text(encoding="utf-8").splitlines()
    domain = infer_domain(file_path)
    title_stack: List[str] = []
    paragraph_buffer: List[str] = []
    chunks: List[DocumentChunk] = []
    chunk_no = 0

    def flush_paragraphs() -> None:
        nonlocal chunk_no
        paragraph = "\n".join(item.strip() for item in paragraph_buffer if item.strip()).strip()
        paragraph_buffer.clear()
        if not paragraph:
            return

        chunk_title = " / ".join(title_stack) if title_stack else file_path.stem
        for piece in split_long_text(paragraph):
            chunk_no += 1
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{file_path.stem}-{chunk_no}",
                    source=file_path.name,
                    domain=domain,
                    title=chunk_title,
                    content=piece,
                )
            )

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            flush_paragraphs()
            continue
        if line.startswith("#"):
            flush_paragraphs()
            level = len(line) - len(line.lstrip("#"))
            heading = line[level:].strip()
            title_stack[:] = title_stack[: level - 1]
            title_stack.append(heading)
            continue
        paragraph_buffer.append(line)

    flush_paragraphs()
    return chunks


def load_chunks_from_dir(docs_dir: Path) -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []
    for file_path in sorted(docs_dir.glob("*.md")):
        chunks.extend(markdown_to_chunks(file_path))
    return chunks
