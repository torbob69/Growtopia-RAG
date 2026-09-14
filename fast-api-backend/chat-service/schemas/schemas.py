from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    # same bounds /search validates, so an over-long query is rejected here rather than
    # one hop later
    query: str = Field(min_length=1, max_length=1000)
    # omit to start a new session; pass one to continue it
    session_id: int | None = None


class ChatResponse(BaseModel):
    session_id: int
    answer: str
    # wiki pages the answer was grounded in, so the caller can check it
    sources: list[str]


class SessionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime


class InteractionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    query: str
    answer: str
    sources: list[str]
    created_at: datetime
