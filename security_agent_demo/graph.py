import json
from pathlib import Path

from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition

from security_agent_demo.agents import (
    AssistantNode,
    image_runnable,
    image_safe_tools,
    ops_runnable,
    ops_safe_tools,
    ops_sensitive_tools,
    primary_runnable,
    primary_tools,
    video_runnable,
    video_safe_tools,
    video_sensitive_tools,
)
from security_agent_demo.config import DATA_DIR
from security_agent_demo.models import CompleteOrEscalate, ToImageAgent, ToOpsAgent, ToVideoAgent
from security_agent_demo.state import AgentState
from security_agent_demo.tools import create_tool_node_with_fallback


def _load_users():
    path = Path(DATA_DIR) / "users.json"
    return json.loads(path.read_text(encoding="utf-8"))


def get_user_info(state: AgentState, config=None):
    configurable = (config or {}).get("configurable", {})
    user_id = configurable.get("user_id", "ops_001")
    users = _load_users()
    user = next((item for item in users if item["user_id"] == user_id), users[0])
    return {
        "user_info": json.dumps(user, ensure_ascii=False),
    }


def create_entry_node(assistant_name: str, new_dialog_state: str):
    def entry_node(state: dict) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    content=(
                        f"现在切换到 {assistant_name}。请回顾之前的对话并继续处理当前任务。"
                        "如果任务已结束或不属于你的职责范围，请调用 CompleteOrEscalate 返回主Agent。"
                    ),
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": new_dialog_state,
        }

    return entry_node


def pop_dialog_state(state: dict) -> dict:
    messages = []
    if state["messages"][-1].tool_calls:
        messages.append(
            ToolMessage(
                content="正在返回主Agent，请结合之前上下文继续服务用户。",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        )
    return {"dialog_state": "pop", "messages": messages}


builder = StateGraph(AgentState)

builder.add_node("fetch_user_info", get_user_info)
builder.add_edge(START, "fetch_user_info")

builder.add_node("primary_agent", AssistantNode(primary_runnable))
builder.add_node("primary_tools", create_tool_node_with_fallback(primary_tools))
builder.add_edge("primary_tools", "primary_agent")

builder.add_node("enter_video_agent", create_entry_node("video_agent", "video_agent"))
builder.add_node("video_agent", AssistantNode(video_runnable))
builder.add_node("video_safe_tools", create_tool_node_with_fallback(video_safe_tools))
builder.add_node("video_sensitive_tools", create_tool_node_with_fallback(video_sensitive_tools))
builder.add_edge("enter_video_agent", "video_agent")
builder.add_edge("video_safe_tools", "video_agent")
builder.add_edge("video_sensitive_tools", "video_agent")

builder.add_node("enter_image_agent", create_entry_node("image_agent", "image_agent"))
builder.add_node("image_agent", AssistantNode(image_runnable))
builder.add_node("image_safe_tools", create_tool_node_with_fallback(image_safe_tools))
builder.add_edge("enter_image_agent", "image_agent")
builder.add_edge("image_safe_tools", "image_agent")

builder.add_node("enter_ops_agent", create_entry_node("ops_agent", "ops_agent"))
builder.add_node("ops_agent", AssistantNode(ops_runnable))
builder.add_node("ops_safe_tools", create_tool_node_with_fallback(ops_safe_tools))
builder.add_node("ops_sensitive_tools", create_tool_node_with_fallback(ops_sensitive_tools))
builder.add_edge("enter_ops_agent", "ops_agent")
builder.add_edge("ops_safe_tools", "ops_agent")
builder.add_edge("ops_sensitive_tools", "ops_agent")

builder.add_node("leave_skill", pop_dialog_state)
builder.add_edge("leave_skill", "primary_agent")


def route_primary_agent(state: dict):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        tool_name = tool_calls[0]["name"]
        if tool_name == ToVideoAgent.__name__:
            return "enter_video_agent"
        if tool_name == ToImageAgent.__name__:
            return "enter_image_agent"
        if tool_name == ToOpsAgent.__name__:
            return "enter_ops_agent"
        return "primary_tools"
    raise ValueError("Invalid route from primary agent.")


builder.add_conditional_edges(
    "primary_agent",
    route_primary_agent,
    ["enter_video_agent", "enter_image_agent", "enter_ops_agent", "primary_tools", END],
)


def route_video_agent(state: dict):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls):
        return "leave_skill"
    safe_names = [tool.name for tool in video_safe_tools]
    if all(tc["name"] in safe_names for tc in tool_calls):
        return "video_safe_tools"
    return "video_sensitive_tools"


builder.add_conditional_edges(
    "video_agent",
    route_video_agent,
    ["video_safe_tools", "video_sensitive_tools", "leave_skill", END],
)


def route_image_agent(state: dict):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls):
        return "leave_skill"
    return "image_safe_tools"


builder.add_conditional_edges(
    "image_agent",
    route_image_agent,
    ["image_safe_tools", "leave_skill", END],
)


def route_ops_agent(state: dict):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls):
        return "leave_skill"
    safe_names = [tool.name for tool in ops_safe_tools]
    if all(tc["name"] in safe_names for tc in tool_calls):
        return "ops_safe_tools"
    return "ops_sensitive_tools"


builder.add_conditional_edges(
    "ops_agent",
    route_ops_agent,
    ["ops_safe_tools", "ops_sensitive_tools", "leave_skill", END],
)


def route_to_workflow(state: dict):
    dialog_state = state.get("dialog_state")
    if not dialog_state:
        return "primary_agent"
    return dialog_state[-1]


builder.add_conditional_edges(
    "fetch_user_info",
    route_to_workflow,
    ["primary_agent", "video_agent", "image_agent", "ops_agent"],
)

memory = MemorySaver()
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["video_sensitive_tools", "ops_sensitive_tools"],
)
