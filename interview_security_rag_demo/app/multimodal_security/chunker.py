from __future__ import annotations

from typing import List

from app.multimodal_security.models import KnowledgeUnit, ParsedAsset


class ChapterSemanticChunker:
    """
    Demo version of:
    chapter splitting + semantic re-chunking + image context binding.
    """

    def __init__(self, max_chars: int = 220):
        self.max_chars = max_chars

    def build_units(self, assets: List[ParsedAsset]) -> List[KnowledgeUnit]:
        units: List[KnowledgeUnit] = []
        for asset in assets:
            domain = self._infer_domain(asset)
            pieces = self._split_text(asset.content)
            if asset.related_media:
                for media in asset.related_media:
                    for index, piece in enumerate(pieces, start=1):
                        units.append(
                            KnowledgeUnit(
                                unit_id=f"{asset.asset_id}-img-{index}-{media}",
                                source=asset.source,
                                modality="text_image",
                                domain=domain,
                                title=asset.title_path,
                                text=piece,
                                image_ref=media,
                            )
                        )
            else:
                for index, piece in enumerate(pieces, start=1):
                    units.append(
                        KnowledgeUnit(
                            unit_id=f"{asset.asset_id}-txt-{index}",
                            source=asset.source,
                            modality="text",
                            domain=domain,
                            title=asset.title_path,
                            text=piece,
                        )
                    )
        return units

    def _infer_domain(self, asset: ParsedAsset) -> str:
        title = asset.title_path
        source = asset.source.lower()
        if "告警" in title or "alarm" in source:
            return "alarm"
        if "部署" in title or "deployment" in source:
            return "deployment"
        if "抓拍" in title or "snapshot" in source:
            return "image"
        return "device"

    def _split_text(self, text: str) -> List[str]:
        if len(text) <= self.max_chars:
            return [text]
        chunks: List[str] = []
        current = ""
        for sentence in text.replace("。", "。\n").splitlines():
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current) + len(sentence) <= self.max_chars:
                current += sentence
            else:
                if current:
                    chunks.append(current)
                current = sentence
        if current:
            chunks.append(current)
        return chunks
