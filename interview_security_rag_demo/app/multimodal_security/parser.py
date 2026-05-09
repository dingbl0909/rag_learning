from __future__ import annotations

from pathlib import Path
from typing import List

from app.multimodal_security.models import ParsedAsset


class SecurityMultimodalParser:
    """
    A readable demo parser that mirrors:
    Dots.OCR / Unstructured / OCR -> normalized assets.

    For understanding purposes, it parses markdown files with simple
    IMAGE_REF / TABLE_REF markers.
    """

    def parse_directory(self, docs_dir: Path) -> List[ParsedAsset]:
        assets: List[ParsedAsset] = []
        for file_path in sorted(docs_dir.glob("*.md")):
            assets.extend(self.parse_file(file_path))
        return assets

    def parse_file(self, file_path: Path) -> List[ParsedAsset]:
        lines = file_path.read_text(encoding="utf-8").splitlines()
        assets: List[ParsedAsset] = []
        title_stack: List[str] = []
        buffer: List[str] = []
        media_refs: List[str] = []
        asset_no = 0

        def flush_block() -> None:
            nonlocal asset_no
            text = "\n".join(line.strip() for line in buffer if line.strip()).strip()
            buffer.clear()
            if not text:
                return
            asset_no += 1
            assets.append(
                ParsedAsset(
                    asset_id=f"{file_path.stem}-asset-{asset_no}",
                    source=file_path.name,
                    asset_type="text_block",
                    title_path=" / ".join(title_stack) if title_stack else file_path.stem,
                    content=text,
                    related_media=list(media_refs),
                )
            )

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                flush_block()
                media_refs.clear()
                continue
            if line.startswith("#"):
                flush_block()
                media_refs.clear()
                level = len(line) - len(line.lstrip("#"))
                heading = line[level:].strip()
                title_stack[:] = title_stack[: level - 1]
                title_stack.append(heading)
                continue
            if line.startswith("IMAGE_REF:"):
                media_refs.append(line.split(":", 1)[1].strip())
                continue
            if line.startswith("TABLE_REF:"):
                media_refs.append(line.split(":", 1)[1].strip())
                continue
            buffer.append(line)

        flush_block()
        return assets
