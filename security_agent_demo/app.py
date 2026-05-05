from fastapi import FastAPI

from security_agent_demo.config import APP_TITLE, APP_VERSION
from security_agent_demo.graph import graph
from security_agent_demo.llm import strip_think_tags
from security_agent_demo.schemas import ChatRequest, ChatResponse, ContinueRequest


app = FastAPI(title=APP_TITLE, version=APP_VERSION)


def build_config(thread_id: str, user_id: str) -> dict:
    return {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
        }
    }


def extract_last_ai_message(events) -> str:
    result = ""
    last_tool_message = ""
    for event in events:
        messages = event.get("messages")
        if not messages:
            continue
        message = messages[-1] if isinstance(messages, list) else messages
        if message.__class__.__name__ == "AIMessage" and message.content:
            result = message.content if isinstance(message.content, str) else str(message.content)
        if message.__class__.__name__ == "ToolMessage" and message.content:
            last_tool_message = message.content if isinstance(message.content, str) else str(message.content)
    if result:
        return strip_think_tags(result)
    return strip_think_tags(last_tool_message)


@app.get("/")
def health():
    return {"status": "ok", "service": APP_TITLE, "version": APP_VERSION}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    config = build_config(req.thread_id, req.user_id)
    events = graph.stream({"messages": ("user", req.message)}, config, stream_mode="values")
    result = extract_last_ai_message(events)
    current_state = graph.get_state(config)
    if bool(current_state.next):
        result = result or "检测到敏感操作，是否批准继续执行？"
    return ChatResponse(
        assistant=result or "已收到请求。",
        thread_id=req.thread_id,
        needs_confirmation=bool(current_state.next),
        next_nodes=list(current_state.next),
    )


@app.post("/chat/continue", response_model=ChatResponse)
def continue_chat(req: ContinueRequest):
    config = build_config(req.thread_id, req.user_id)
    if req.approve:
        events = graph.stream(None, config, stream_mode="values")
    else:
        user_message = req.message or "我暂不批准，请给我新的建议。"
        events = graph.stream({"messages": ("user", user_message)}, config, stream_mode="values")
    result = extract_last_ai_message(events)
    current_state = graph.get_state(config)
    if bool(current_state.next):
        result = result or "流程仍停留在敏感操作确认阶段，请继续确认或补充说明。"
    return ChatResponse(
        assistant=result or "流程已继续。",
        thread_id=req.thread_id,
        needs_confirmation=bool(current_state.next),
        next_nodes=list(current_state.next),
    )
