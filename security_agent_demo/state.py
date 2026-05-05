from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


DialogState = Literal["primary_agent", "video_agent", "image_agent", "ops_agent"]


def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str
    dialog_state: Annotated[list[DialogState], update_dialog_stack]
