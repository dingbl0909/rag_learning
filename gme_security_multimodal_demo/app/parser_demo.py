from __future__ import annotations

from pathlib import Path
from typing import List

from app.models import ParsedBlock


class SecurityDocumentParserDemo:
    """
    Demo parser for Dots.OCR + layout extraction.

    Markdown markers:
    - IMAGE_REF:   screenshot / figure id
    - TABLE_REF:   table id (structured block)
    - BASE64_REF:  simulated image payload id from OCR pipeline
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
        image_refs: List[str] = []
        table_refs: List[str] = []
        base64_refs: List[str] = []
        blocks: List[ParsedBlock] = []
        block_no = 0

        def flush() -> None:
            nonlocal block_no
            text = "\n".join(line.strip() for line in buffer if line.strip()).strip()
            buffer.clear()
            if not text and not image_refs and not table_refs:
                return
            block_no += 1
            blocks.append(
                ParsedBlock(
                    block_id=f"{file_path.stem}-block-{block_no}",
                    source=file_path.name,
                    title_path=" / ".join(title_stack) if title_stack else file_path.stem,
                    text=text,
                    image_refs=list(image_refs),
                    table_refs=list(table_refs),
                    base64_refs=list(base64_refs),
                )
            )

        for raw in lines:
            line = raw.strip()
            if not line:
                flush()
                image_refs.clear()
                table_refs.clear()
                base64_refs.clear()
                continue
            if line.startswith("#"):
                flush()
                image_refs.clear()
                table_refs.clear()
                base64_refs.clear()
                level = len(line) - len(line.lstrip("#"))
                heading = line[level:].strip()
                title_stack[:] = title_stack[: level - 1]
                title_stack.append(heading)
                continue
            if line.startswith("IMAGE_REF:"):
                image_refs.append(line.split(":", 1)[1].strip())
                continue
            if line.startswith("TABLE_REF:"):
                table_refs.append(line.split(":", 1)[1].strip())
                continue
            if line.startswith("BASE64_REF:"):
                base64_refs.append(line.split(":", 1)[1].strip())
                continue
            buffer.append(line)

        flush()
        return blocks
