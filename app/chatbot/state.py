"""
Typed state object for the LangGraph chatbot workflow.
"""
from typing import Annotated, Optional
from typing_extensions import TypedDict


class ChatState(TypedDict):
    """Shared state passed between LangGraph nodes."""

    session_id: str
    question: str
    chat_history: list[dict]          # [{role, content}, ...]
    rewritten_query: str
    retrieved_documents: list          # list[Document]
    context: str                       # formatted context string
    answer: str
    sources: list[dict]                # [{source, page, chunk_index}, ...]
    is_relevant: bool                  # True if retrieved docs pass threshold
    error: Optional[str]               # non-None when a node fails
    messages: list                     # LangChain BaseMessage objects for the run
    tool_calls: list                   # Recorded tool calls from orchestrator
