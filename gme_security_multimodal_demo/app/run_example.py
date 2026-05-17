from __future__ import annotations

import json
from pathlib import Path

from app.workflow_graph import MultimodalSecurityGraph


def build_demo() -> MultimodalSecurityGraph:
    docs_dir = Path(__file__).resolve().parent.parent / "data" / "security_multimodal_docs"
    return MultimodalSecurityGraph(docs_dir)


def run_all_examples() -> dict:
    demo = build_demo()
    cases = {
        "text_image_alarm": demo.ask(
            user_id="demo_user",
            question="用户上传了一张布控告警时间线截图，想知道误报应该怎么排查？",
            image_ref="alarm_timeline_screenshot_01",
        ),
        "text_only_deployment": demo.ask(
            user_id="demo_user",
            question="Milvus 部署后检索不到结果，应该怎么排查？",
        ),
        "image_only_snapshot": demo.ask(
            user_id="demo_user_2",
            question="",
            image_ref="gate_snapshot_01",
        ),
    }
    cases["cache_hit_repeat"] = demo.ask(
        user_id="demo_user",
        question="Milvus 部署后检索不到结果，应该怎么排查？",
    )
    return {name: result.model_dump() for name, result in cases.items()}


def run_example() -> dict:
    return run_all_examples()["text_image_alarm"]


if __name__ == "__main__":
    print(json.dumps(run_all_examples(), ensure_ascii=False, indent=2))
