from __future__ import annotations

import json
from pathlib import Path

from app.workflow_demo import MultimodalSecurityWorkflowDemo


def build_demo() -> MultimodalSecurityWorkflowDemo:
    docs_dir = Path(__file__).resolve().parent.parent / "data" / "security_multimodal_docs"
    return MultimodalSecurityWorkflowDemo(docs_dir)


def run_example() -> dict:
    demo = build_demo()
    result = demo.ask(
        user_id="demo_user",
        question="用户上传了一张布控告警时间线截图，想知道误报应该怎么排查？",
        image_ref="alarm_timeline_screenshot_01",
    )
    return result.model_dump()


if __name__ == "__main__":
    print(json.dumps(run_example(), ensure_ascii=False, indent=2))
