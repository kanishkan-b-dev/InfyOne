"""
LangGraph workflow graph for the Athena chatbot.

Graph topology:
    START
      ↓
    load_session_history
      ↓
    orchestrator (LLM decides: Tool Call or Direct Answer)
      ├─ [Has Tool Calls] ──→ execute_tools ──→ generate_answer ──→ update_session ──→ END
      └─ [Direct Answer]  ────────────────────────────────────────→ update_session ──→ END
"""
import logging
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.chatbot.nodes import (
    execute_tools,
    generate_answer,
    load_session_history,
    orchestrator,
    route_orchestrator,
    update_session,
)
from app.chatbot.state import ChatState

logger = logging.getLogger(__name__)


@lru_cache()
def build_graph():
    """Compile and cache the LangGraph orchestrator workflow."""
    graph = StateGraph(ChatState)

    # ── Register nodes ──────────────────────────────────────────
    graph.add_node("load_session_history", load_session_history)
    graph.add_node("orchestrator", orchestrator)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("update_session", update_session)

    # ── Wire edges ──────────────────────────────────────────────
    graph.add_edge(START, "load_session_history")
    graph.add_edge("load_session_history", "orchestrator")

    graph.add_conditional_edges(
        "orchestrator",
        route_orchestrator,
        {
            "execute_tools": "execute_tools",
            "update_session": "update_session",
        },
    )

    graph.add_edge("execute_tools", "generate_answer")
    graph.add_edge("generate_answer", "update_session")
    graph.add_edge("update_session", END)

    compiled = graph.compile()
    logger.info("LangGraph workflow compiled successfully.")
    return compiled


async def run_chat(session_id: str, question: str) -> dict:
    """
    Execute the chatbot graph for a given session and question.

    Returns a dict with keys: answer, sources, rewritten_query.
    """
    app_graph = build_graph()

    initial_state: ChatState = {
        "session_id": session_id,
        "question": question,
        "chat_history": [],
        "rewritten_query": "",
        "retrieved_documents": [],
        "context": "",
        "answer": "",
        "sources": [],
        "is_relevant": False,
        "error": None,
        "messages": [],
        "tool_calls": [],
    }

    final_state = await app_graph.ainvoke(initial_state)

    return {
        "answer": final_state.get("answer", ""),
        "sources": final_state.get("sources", []),
        "rewritten_query": final_state.get("rewritten_query", question),
    }

