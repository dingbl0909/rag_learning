from __future__ import annotations

from pathlib import Path

from app.multimodal_security.workflow import MultimodalSecurityRAGDemo


def build_demo() -> MultimodalSecurityRAGDemo:
    docs_dir = Path(__file__).resolve().parent.parent.parent / "data" / "multimodal_security_docs"
    return MultimodalSecurityRAGDemo(docs_dir=docs_dir)


def run_example() -> dict:
    demo = build_demo()
    result = demo.ask(
        user_id="interview_user",
        question="用户上传了一张告警时间线截图，想知道误报应该如何排查？",
        image_ref="alarm_timeline_snapshot_01",
    )
    return result.model_dump()


if __name__ == "__main__":
    import json

    print(json.dumps(run_example(), ensure_ascii=False, indent=2))
