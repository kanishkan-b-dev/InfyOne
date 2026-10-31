"""
LangGraph workflow graph for the Infosys AI Chatbot.

Graph topology:
    START
      ↓
    reset_turn_state (State hygiene: resets transient variables)
      ↓
    classify_intent
      ├─ [greeting_smalltalk]  ──→ respond_smalltalk ──────────────────→ END
      ├─ [meta_bot]            ──→ respond_meta_bot ────────────────────→ END
      ├─ [chat_history]        ──→ respond_from_history ──────────────→ END
      ├─ [out_of_scope]        ──→ respond_out_of_scope ──────────────→ END
      ├─ [infosys_question]    ──→ retrieve ────────→ generate_answer ─→ END
      └─ [follow_up]           ──→ rewrite_query ───→ retrieve ────────→ generate_answer ─→ END
"""
import logging
from typing import Optional

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.chatbot.nodes import (
    classify_intent,
    generate_answer,
    reset_turn_state,
    respond_from_history,
    respond_meta_bot,
    respond_out_of_scope,
    respond_smalltalk,
    retrieve,
    rewrite_query,
)
from app.chatbot.state import ChatState

logger = logging.getLogger(__name__)

# Module-level checkpointer instance (MemorySaver by default; easily swappable for SqliteSaver/Postgres)
_default_checkpointer = MemorySaver()
_compiled_graph = None


def get_checkpointer() -> BaseCheckpointSaver:
    """Return the active checkpointer. Swap this implementation for SqliteSaver or Postgres."""
    return _default_checkpointer


def route_by_intent(state: ChatState) -> str:
    """Conditional edge routing from classify_intent based on detected intent."""
    intent = state.get("intent", "infosys_question")
    if intent == "greeting_smalltalk":
        return "respond_smalltalk"
    elif intent == "meta_bot":
        return "respond_meta_bot"
    elif intent == "chat_history":
        return "respond_from_history"
    elif intent == "follow_up":
        return "rewrite_query"
    elif intent == "out_of_scope":
        return "respond_out_of_scope"
    else:  # "infosys_question"
        return "retrieve"


def build_graph(checkpointer: Optional[BaseCheckpointSaver] = None):
    """
    Construct, wire, and compile the LangGraph StateGraph with checkpointing.
    """
    builder = StateGraph(ChatState)

    # 1. Register nodes
    builder.add_node("reset_turn_state", reset_turn_state)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("respond_smalltalk", respond_smalltalk)
    builder.add_node("respond_meta_bot", respond_meta_bot)
    builder.add_node("respond_from_history", respond_from_history)
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("respond_out_of_scope", respond_out_of_scope)

    # 2. Wire edges: START -> reset_turn_state -> classify_intent
    builder.add_edge(START, "reset_turn_state")
    builder.add_edge("reset_turn_state", "classify_intent")

    builder.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "respond_smalltalk": "respond_smalltalk",
            "respond_meta_bot": "respond_meta_bot",
            "respond_from_history": "respond_from_history",
            "rewrite_query": "rewrite_query",
            "retrieve": "retrieve",
            "respond_out_of_scope": "respond_out_of_scope",
        },
    )

    # Follow-up branch passes through rewrite_query then retrieve
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("retrieve", "generate_answer")

    # All answer nodes route to END
    builder.add_edge("respond_smalltalk", END)
    builder.add_edge("respond_meta_bot", END)
    builder.add_edge("respond_from_history", END)
    builder.add_edge("generate_answer", END)
    builder.add_edge("respond_out_of_scope", END)

    saver = checkpointer if checkpointer is not None else get_checkpointer()
    compiled = builder.compile(checkpointer=saver)
    logger.info("LangGraph workflow compiled successfully with %s.", saver.__class__.__name__)
    return compiled


def get_graph():
    """Return the singleton compiled graph instance."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


async def run_chat(session_id: str, question: str) -> dict:
    """
    Execute the chat workflow for a given session and question.

    - Uses checkpointer with config thread_id=session_id.
    - Sends ONLY the new user message each turn; checkpointer restores the full history.
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}

    input_data = {
        "messages": [HumanMessage(content=question)],
    }

    final_state = await graph.ainvoke(input_data, config=config)

    answer = final_state.get("answer", "")
    sources = final_state.get("sources", [])
    rewritten_query = final_state.get("rewritten_query")
    intent = final_state.get("intent")
    answering_node = final_state.get("answering_node")

    logger.info(
        "[run_chat] session=%s | intent=%s | answering_node=%s | docs=%d",
        session_id,
        intent,
        answering_node,
        len(final_state.get("documents", [])),
    )

    return {
        "session_id": session_id,
        "answer": answer,
        "sources": sources,
        "rewritten_query": rewritten_query,
        "intent": intent,
        "answering_node": answering_node,
    }
