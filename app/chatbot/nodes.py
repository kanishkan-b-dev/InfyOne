"""
LangGraph node implementations for the Athena chatbot workflow.
Each node is a pure async function: ChatState → partial ChatState update.
"""
import logging

import json
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import PromptTemplate

from app.chatbot.prompts import (
    FALLBACK_PROMPT,
    QUERY_REWRITE_PROMPT,
    RAG_ANSWER_PROMPT,
    SYSTEM_PROMPT,
)
from app.chatbot.state import ChatState
from app.chatbot.tools import execute_knowledge_search, search_athena_knowledge_base
from app.llm.provider import get_llm
from app.memory.redis_manager import redis_manager
from app.rag.retriever import format_context, retrieve_documents

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_history(history: list[dict]) -> str:
    """Convert [{role, content}] to a readable string for prompts."""
    if not history:
        return "No previous conversation."
    lines = []
    for msg in history[-10:]:  # last 5 exchanges (10 messages)
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def _history_to_lc_messages(history: list[dict]) -> list:
    """Convert history dicts to LangChain message objects."""
    msgs = []
    for msg in history:
        if msg["role"] == "user":
            msgs.append(HumanMessage(content=msg["content"]))
        else:
            msgs.append(AIMessage(content=msg["content"]))
    return msgs


# ── Node: load_session_history ────────────────────────────────────────────────

async def load_session_history(state: ChatState) -> dict:
    """Load conversation history from Redis."""
    session_id = state["session_id"]
    try:
        history = await redis_manager.get_history(session_id)
        logger.info("[node:load_history] session=%s messages=%d", session_id, len(history))
    except Exception as exc:
        logger.warning("[node:load_history] Redis unavailable: %s — using empty history.", exc)
        history = []
    return {"chat_history": history}


def _is_simple_greeting(text: str) -> bool:
    """Check if input is purely a greeting, farewell, or pleasantry."""
    import re
    clean = re.sub(r"[^\w\s]", "", text.strip().lower()).strip()
    greetings = {
        "hi", "hello", "hey", "hola", "namaste", "good morning", "good afternoon",
        "good evening", "how are you", "who are you", "what can you do", "help",
        "thanks", "thank you", "thank you so much", "thx", "bye", "goodbye", "see you",
        "ok", "okay", "cool", "great", "welcome"
    }
    if clean in greetings:
        return True
    words = clean.split()
    if len(words) <= 2 and all(w in {"hi", "hello", "hey", "there", "thanks", "bye", "good", "morning", "afternoon", "evening"} for w in words):
        return True
    return False


# ── Node: orchestrator (Tool-calling routing layer) ───────────────────────────

async def orchestrator(state: ChatState) -> dict:
    """
    LLM Orchestrator: decides whether to answer directly (greetings/general conversation)
    or invoke tools for RAG knowledge search.
    """
    question = state["question"]
    history = state.get("chat_history", [])

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(_history_to_lc_messages(history[-6:]))  # last 3 exchanges
    messages.append(HumanMessage(content=question))

    try:
        llm = get_llm()
        llm_with_tools = llm.bind_tools([search_athena_knowledge_base])
        response = await llm_with_tools.ainvoke(messages)

        tool_calls = getattr(response, "tool_calls", []) or []

        # If LLM didn't call a tool, but user asked a non-greeting question,
        # enforce search so we don't return false negative refusals
        if not tool_calls and not _is_simple_greeting(question):
            import uuid
            tool_id = f"call_{uuid.uuid4().hex[:12]}"
            logger.info(
                "[node:orchestrator] Non-greeting question without tool call; enforcing knowledge search: '%s' (id=%s)",
                question[:50],
                tool_id,
            )
            synthetic_tool_call = {
                "name": "search_athena_knowledge_base",
                "args": {"query": question},
                "id": tool_id,
                "type": "tool_call",
            }
            tool_calls = [synthetic_tool_call]
            response = AIMessage(
                content="",
                tool_calls=tool_calls,
            )

        if tool_calls:
            logger.info(
                "[node:orchestrator] Tool call triggered: %s",
                [tc.get("name") for tc in tool_calls],
            )
            return {
                "messages": [*messages, response],
                "tool_calls": tool_calls,
                "rewritten_query": tool_calls[0].get("args", {}).get("query", question),
            }
        else:
            # Direct response without tool invocation (greeting, small talk, general capability)
            answer = response.content.strip() if isinstance(response.content, str) else str(response.content)
            logger.info("[node:orchestrator] Direct response generated (no tool call needed).")
            return {
                "messages": [*messages, response],
                "tool_calls": [],
                "answer": answer,
                "sources": [],
                "is_relevant": False,
                "error": None,
            }

    except Exception as exc:
        logger.error("[node:orchestrator] Orchestrator call failed: %s", exc)
        # Fallback direct response
        return {
            "messages": messages,
            "tool_calls": [],
            "answer": "Hello! I am the official AI Assistant for Athena Institute of Technology. How can I assist you today?",
            "sources": [],
            "error": str(exc),
        }


# ── Conditional Edge: route_orchestrator ──────────────────────────────────────

def route_orchestrator(state: ChatState) -> str:
    """Route to 'execute_tools' if tools were called, otherwise 'update_session'."""
    tool_calls = state.get("tool_calls", [])
    if tool_calls:
        return "execute_tools"
    return "update_session"


# ── Node: execute_tools (Runs knowledge base retrieval) ───────────────────────

async def execute_tools(state: ChatState) -> dict:
    """Execute tool calls generated by the orchestrator and record citations/sources."""
    tool_calls = state.get("tool_calls", [])
    current_messages = list(state.get("messages", []))

    all_docs = []
    all_sources = []
    contexts = []

    for tool_call in tool_calls:
        tool_name = tool_call.get("name")
        args = tool_call.get("args", {})
        query = args.get("query", state["question"])
        tool_id = tool_call.get("id", "tool_call_0")

        if tool_name == "search_athena_knowledge_base" or "knowledge" in tool_name.lower():
            logger.info("[node:execute_tools] Executing retrieval for query='%s...'", query[:50])
            context, docs, sources = execute_knowledge_search(query)
            all_docs.extend(docs)
            all_sources.extend(sources)
            contexts.append(context)

            current_messages.append(
                ToolMessage(
                    content=context if context else "No relevant documents found.",
                    tool_call_id=tool_id,
                )
            )
        else:
            current_messages.append(
                ToolMessage(
                    content=f"Unknown tool '{tool_name}'.",
                    tool_call_id=tool_id,
                )
            )

    combined_context = "\n\n---\n\n".join(contexts)
    is_relevant = bool(all_docs)

    logger.info(
        "[node:execute_tools] Completed. Docs retrieved=%d, Sources=%d",
        len(all_docs),
        len(all_sources),
    )

    return {
        "messages": current_messages,
        "retrieved_documents": all_docs,
        "context": combined_context,
        "sources": all_sources,
        "is_relevant": is_relevant,
    }


# ── Node: generate_answer (Synthesizes answer from tool outputs) ─────────────

async def generate_answer(state: ChatState) -> dict:
    """Call the LLM with tool results to generate the final grounded response."""
    messages = state.get("messages", [])

    try:
        llm = get_llm()
        response = await llm.ainvoke(messages)
        answer = response.content.strip() if isinstance(response.content, str) else str(response.content)
        logger.info("[node:generate_answer] Final synthesized answer length=%d", len(answer))
        return {"answer": answer, "error": None}

    except Exception as exc:
        logger.error("[node:generate_answer] LLM call failed: %s", exc)
        return {
            "answer": "I encountered an issue generating the final response. Please try again.",
            "error": str(exc),
        }


# ── Legacy & Unit-Test Support Nodes ──────────────────────────────────────────

async def rewrite_query(state: ChatState) -> dict:
    """Rewrite query (used for test suite compatibility)."""
    question = state["question"]
    history = state.get("chat_history", [])

    if not history:
        return {"rewritten_query": question}

    try:
        llm = get_llm()
        prompt = QUERY_REWRITE_PROMPT.format(
            history=_format_history(history),
            question=question,
        )
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        rewritten = response.content.strip()
        return {"rewritten_query": rewritten if rewritten else question}
    except Exception:
        return {"rewritten_query": question}


async def retrieve_context(state: ChatState) -> dict:
    """Direct context retrieval (used for test suite compatibility)."""
    import asyncio
    query = state.get("rewritten_query") or state["question"]
    try:
        docs = await asyncio.to_thread(retrieve_documents, query)
        context = format_context(docs)
        sources = [
            {
                "source": d.metadata.get("title") or d.metadata.get("source", "Unknown"),
                "title": d.metadata.get("title", ""),
                "category": d.metadata.get("category", ""),
                "page": d.metadata.get("page"),
                "chunk_index": d.metadata.get("chunk_index"),
            }
            for d in docs
        ]
        return {
            "retrieved_documents": docs,
            "context": context,
            "is_relevant": bool(docs),
            "sources": sources,
        }
    except Exception as exc:
        return {"retrieved_documents": [], "context": "", "is_relevant": False, "sources": []}


async def fallback_response(state: ChatState) -> dict:
    """Context-aware fallback response."""
    question = state.get("rewritten_query") or state["question"]
    history = state.get("chat_history", [])
    try:
        llm = get_llm()
        prompt = FALLBACK_PROMPT.format(
            question=question,
            history=_format_history(history),
        )
        response = await llm.ainvoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])
        answer = response.content.strip()
    except Exception:
        answer = (
            "Hello! I am the official AI Assistant for Athena Institute of Technology. "
            "How can I help you today with admissions, courses, fees, or campus facilities?"
        )
    return {"answer": answer, "sources": [], "error": None}


def route_by_relevance(state: ChatState) -> str:
    """Route by relevance flag."""
    return "generate_answer" if state.get("is_relevant") else "fallback_response"


# ── Node: update_session ──────────────────────────────────────────────────────

async def update_session(state: ChatState) -> dict:
    """Persist the latest exchange to Redis."""
    try:
        await redis_manager.append_messages(
            session_id=state["session_id"],
            user_msg=state["question"],
            assistant_msg=state["answer"],
        )
        logger.info("[node:update_session] session=%s updated.", state["session_id"])
    except Exception as exc:
        logger.warning("[node:update_session] Redis write failed: %s", exc)
    return {}

