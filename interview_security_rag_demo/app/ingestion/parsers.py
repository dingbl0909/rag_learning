from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class ParsedPage:
    page_no: int
    text_blocks: List[str] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)


@dataclass
class ParsedDocument:
    source_path: str
    source_type: str
    parser_name: str
    pages: List[ParsedPage]
    metadata: Dict[str, str] = field(default_factory=dict)


class PyMuPDFParserAdapter:
    """Blueprint for PDF/PPT parsing based on PyMuPDF."""

    parser_name = "PyMuPDFParserAdapter"

    def parse(self, file_path: str) -> ParsedDocument:
        suffix = Path(file_path).suffix.lower()
        return ParsedDocument(
            source_path=file_path,
            source_type=suffix or "unknown",
            parser_name=self.parser_name,
            pages=[
                ParsedPage(
                    page_no=1,
                    text_blocks=[
                        "Mock page content extracted by PyMuPDF. In production this would preserve reading order and layout hints."
                    ],
                    tables=["<table><tr><td>table placeholder</td></tr></table>"],
                    images=["page_1_image_1.png"],
                )
            ],
            metadata={"status": "blueprint", "note": "Replace with fitz.Document parsing in production."},
        )


class UnstructuredHiResLoaderAdapter:
    """Blueprint for loading Word/HTML/PDF using the hi_res strategy."""

    loader_name = "UnstructuredHiResLoaderAdapter"

    def load(self, file_path: str) -> ParsedDocument:
        return ParsedDocument(
            source_path=file_path,
            source_type=Path(file_path).suffix.lower() or "unknown",
            parser_name=self.loader_name,
            pages=[
                ParsedPage(
                    page_no=1,
                    text_blocks=[
                        "Mock high-resolution layout extraction. Intended to preserve headings, tables, and complex page structures."
                    ]
                )
            ],
            metadata={"strategy": "hi_res", "status": "blueprint"},
        )


class TesseractOCRAdapter:
    """Blueprint for OCR extraction from screenshots and scanned pages."""

    engine_name = "TesseractOCRAdapter"

    def extract_text(self, image_path: str) -> Dict[str, str]:
        return {
            "image_path": image_path,
            "engine": self.engine_name,
            "ocr_text": "Mock OCR result extracted from screenshot or scanned page.",
            "status": "blueprint",
        }
