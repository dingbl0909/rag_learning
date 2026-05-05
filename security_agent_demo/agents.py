from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig

from security_agent_demo.llm import llm
from security_agent_demo.models import CompleteOrEscalate, ToImageAgent, ToOpsAgent, ToVideoAgent
from security_agent_demo.state import AgentState
from security_agent_demo.tools import (
    close_alert,
    create_work_order,
    describe_image_issue,
    get_camera_status,
    locate_video_clip,
    request_device_restart,
    search_knowledge_base,
    search_snapshot_records,
    search_video_events,
)


class AssistantNode:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: AgentState, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state, config=config)
            content = result.content
            if not result.tool_calls and (not content or (isinstance(content, list) and not content[0].get("text"))):
                state = {**state, "messages": state["messages"] + [("user", "请直接给出明确答复。")]}
                continue
            return {"messages": result}


primary_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是安防智能助手的主调度Agent，负责理解用户问题并把任务分发给合适的子Agent。"
            "如果问题涉及视频流告警、录像检索、摄像头状态，请优先委派给 video_agent。"
            "如果问题涉及抓拍图片、图像分析、图片取证，请优先委派给 image_agent。"
            "如果问题涉及运维SOP、工单建议、排障指引，请优先委派给 ops_agent。"
            "你可以先回答简单问题，但不要输出<think>或任何思维链，只输出最终答复。"
            "\n当前用户信息：{user_info}"
            "\n当前时间：{time}",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

primary_tools = [search_knowledge_base]
primary_runnable = primary_prompt | llm.bind_tools(
    primary_tools + [ToVideoAgent, ToImageAgent, ToOpsAgent]
)


video_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是 video_agent，专门处理视频流告警、录像检索、摄像头状态和事件回溯。"
            "优先用工具查证，再给用户结论。"
            "如果需要关闭告警或重启设备，这属于敏感操作，先整理理由，再调用工具。"
            "如果问题不属于视频流，或者用户改变需求，请调用 CompleteOrEscalate。"
            "不要输出<think>，只输出最终答复。"
            "\n当前用户信息：{user_info}"
            "\n当前时间：{time}",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

video_safe_tools = [get_camera_status, search_video_events, locate_video_clip]
video_sensitive_tools = [close_alert, request_device_restart]
video_runnable = video_prompt | llm.bind_tools(video_safe_tools + video_sensitive_tools + [CompleteOrEscalate])


image_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是 image_agent，专门处理图片抓拍、图像分析和图片取证问答。"
            "优先使用工具获取抓拍和分析结果，再进行解释。"
            "如果问题不属于图片流，请调用 CompleteOrEscalate。"
            "不要输出<think>，只输出最终答复。"
            "\n当前用户信息：{user_info}"
            "\n当前时间：{time}",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

image_safe_tools = [search_snapshot_records, describe_image_issue]
image_sensitive_tools: list = []
image_runnable = image_prompt | llm.bind_tools(image_safe_tools + [CompleteOrEscalate])


ops_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是 ops_agent，专门处理安防运维、SOP问答、工单建议和标准操作指引。"
            "优先查询知识库，再给出结构化建议。"
            "如果需要创建工单，这是敏感操作，需要先明确原因后再调用工具。"
            "如果问题不属于运维知识或SOP，请调用 CompleteOrEscalate。"
            "不要输出<think>，只输出最终答复。"
            "\n当前用户信息：{user_info}"
            "\n当前时间：{time}",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

ops_safe_tools = [search_knowledge_base]
ops_sensitive_tools = [create_work_order]
ops_runnable = ops_prompt | llm.bind_tools(ops_safe_tools + ops_sensitive_tools + [CompleteOrEscalate])
