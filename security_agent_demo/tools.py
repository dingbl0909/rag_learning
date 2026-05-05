import json
from pathlib import Path
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from security_agent_demo.config import DATA_DIR


def _load_json(name: str) -> list[dict[str, Any]]:
    path = Path(DATA_DIR) / name
    return json.loads(path.read_text(encoding="utf-8"))


def _format_records(title: str, records: list[dict[str, Any]]) -> str:
    if not records:
        return f"{title}：未找到相关记录。"
    lines = [title]
    for record in records:
        lines.append(json.dumps(record, ensure_ascii=False))
    return "\n".join(lines)


@tool
def get_camera_status(camera_id: str = "", location: str = "") -> str:
    """Query a camera status by camera ID or location."""
    records = _load_json("cameras.json")
    matched = [
        item
        for item in records
        if (camera_id and camera_id in item["camera_id"]) or (location and location in item["location"])
    ]
    return _format_records("摄像头状态查询结果", matched)


@tool
def search_video_events(keyword: str) -> str:
    """Search video alert events by keyword, event type, or location."""
    records = _load_json("video_events.json")
    matched = [
        item
        for item in records
        if keyword in item["event_type"] or keyword in item["location"] or keyword in item["summary"]
    ]
    return _format_records("视频事件检索结果", matched)


@tool
def locate_video_clip(event_id: str) -> str:
    """Locate the clip path for a known video event ID."""
    records = _load_json("video_events.json")
    matched = [item for item in records if event_id == item["event_id"]]
    return _format_records("录像片段定位结果", matched)


@tool
def search_snapshot_records(keyword: str) -> str:
    """Search image snapshot records by keyword or camera."""
    records = _load_json("snapshots.json")
    matched = [
        item
        for item in records
        if keyword in item["description"] or keyword in item["analysis"] or keyword in item["camera_id"]
    ]
    return _format_records("图片抓拍检索结果", matched)


@tool
def describe_image_issue(snapshot_id: str) -> str:
    """Return the description and analysis of a snapshot."""
    records = _load_json("snapshots.json")
    matched = [item for item in records if snapshot_id == item["snapshot_id"]]
    return _format_records("图片分析说明", matched)


@tool
def search_knowledge_base(question: str) -> str:
    """Search the built-in security knowledge base for SOP or troubleshooting guidance."""
    records = _load_json("knowledge_base.json")
    matched = [item for item in records if question in item["title"] or question in item["content"]]
    if not matched:
        matched = records[:2]
    return _format_records("知识库检索结果", matched)


@tool
def create_work_order(title: str, detail: str) -> str:
    """Create a work order for manual follow-up. This is a sensitive action."""
    return json.dumps(
        {
            "status": "created",
            "work_order_id": "wo_demo_001",
            "title": title,
            "detail": detail,
        },
        ensure_ascii=False,
    )


@tool
def close_alert(event_id: str, reason: str) -> str:
    """Close a known alert after manual confirmation. This is a sensitive action."""
    return json.dumps(
        {
            "status": "closed",
            "event_id": event_id,
            "reason": reason,
        },
        ensure_ascii=False,
    )


@tool
def request_device_restart(camera_id: str, reason: str) -> str:
    """Request a remote restart for a camera. This is a sensitive action."""
    return json.dumps(
        {
            "status": "restart_requested",
            "camera_id": camera_id,
            "reason": reason,
        },
        ensure_ascii=False,
    )


def handle_tool_error(state: dict) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(content=f"工具调用失败：{repr(error)}", tool_call_id=tc["id"])
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list):
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)],
        exception_key="error",
    )
