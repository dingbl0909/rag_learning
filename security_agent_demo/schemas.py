from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(description="User input message.")
    thread_id: str = Field(description="Conversation thread ID.")
    user_id: str = Field(default="ops_001", description="Mock user ID.")


class ContinueRequest(BaseModel):
    thread_id: str = Field(description="Conversation thread ID.")
    user_id: str = Field(default="ops_001", description="Mock user ID.")
    approve: bool = Field(default=True, description="Whether to continue the interrupted action.")
    message: str | None = Field(default=None, description="Optional clarification if not approving.")


class ChatResponse(BaseModel):
    assistant: str
    thread_id: str
    needs_confirmation: bool = False
    next_nodes: list[str] = []
