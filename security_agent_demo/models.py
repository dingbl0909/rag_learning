from pydantic import BaseModel, Field


class CompleteOrEscalate(BaseModel):
    cancel: bool = True
    reason: str = Field(description="Explain why the task should return to the primary agent.")


class ToVideoAgent(BaseModel):
    request: str = Field(description="Video stream question, alert analysis, or clip retrieval request.")


class ToImageAgent(BaseModel):
    request: str = Field(description="Image stream question, snapshot analysis, or image evidence request.")


class ToOpsAgent(BaseModel):
    request: str = Field(description="Operations question, SOP lookup, work order, or troubleshooting request.")
