from __future__ import annotations

from pathlib import Path
from typing import List

from app.models import ParsedBlock


class SecurityDocumentParserDemo:
    """
    Demo parser for:
    Dots.OCR + OCR + document layout extraction.

    It reads markdown files with IMAGE_REF / TABLE_REF markers and converts them
    into normalized parsed blocks.
    """

    def parse_dir(self, docs_dir: Path) -> List[ParsedBlock]:
        blocks: List[ParsedBlock] = []
        for file_path in sorted(docs_dir.glob("*.md")):
            blocks.extend(self.parse_file(file_path))
        return blocks

    def parse_file(self, file_path: Path) -> List[ParsedBlock]:
        lines = file_path.read_text(encoding="utf-8").splitlines()
        title_stack: List[str] = []
        buffer: List[str] = []
        media_refs: List[str] = []
        blocks: List[ParsedBlock] = []
        block_no = 0

        def flush() -> None:
            nonlocal block_no
            text = "\n".join(line.strip() for line in buffer if line.strip()).strip()
            buffer.clear()
            if not text:
                return
            block_no += 1
            blocks.append(
                ParsedBlock(
                    block_id=f"{file_path.stem}-block-{block_no}",
                    source=file_path.name,
                    title_path=" / ".join(title_stack) if title_stack else file_path.stem,
                    text=text,
                    media_refs=list(media_refs),
                )
            )

        for raw in lines:
            line = raw.strip()
            if not line:
                flush()
                media_refs.clear()
                continue
            if line.startswith("#"):
                flush()
                media_refs.clear()
                level = len(line) - len(line.lstrip("#"))
                heading = line[level:].strip()
                title_stack[:] = title_stack[: level - 1]
                title_stack.append(heading)
                continue
            if line.startswith("IMAGE_REF:") or line.startswith("TABLE_REF:"):
                media_refs.append(line.split(":", 1)[1].strip())
                continue
            buffer.append(line)

        flush()
        return blocks
