"""
Pydantic request/response schemas for the chat API.
"""
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128, description="Unique session identifier")
    message: str = Field(..., min_length=1, max_length=2000, description="User message")

    @field_validator("message")
    @classmethod
    def strip_message(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Message cannot be empty or whitespace only.")
        return stripped


class SourceDoc(BaseModel):
    source: str
    page: Optional[int] = None
    chunk_index: Optional[int] = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceDoc] = []
    rewritten_query: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    redis: str
    chromadb: str
    llm: str
    version: str
