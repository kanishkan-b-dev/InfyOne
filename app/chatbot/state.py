"""
Typed state object for the LangGraph chatbot workflow.
"""
from typing import Annotated, Optional
from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    """
    Shared state for the LangGraph chat workflow.
    - messages uses add_messages reducer so LangGraph checkpointer maintains full history.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    intent: str
    rewritten_query: str
    category_filter: Optional[str]
    documents: list[dict]
    answer: str
    sources: list[dict]
    answering_node: str
