"""
FastAPI router for the /api/chat and /health endpoints.
"""
import logging
import time
import uuid

from fastapi import APIRouter, HTTPException, Request, status

from app.chatbot.graph import run_chat
from app.core.config import get_settings
from app.core.logging import log_conversation
from app.memory.redis_manager import redis_manager
from app.schemas.chat import ChatRequest, ChatResponse, HealthResponse, SourceDoc
from app.services.chromadb_service import get_chroma_service_v2

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()


@router.post("/api/chat", response_model=ChatResponse, summary="Send a message to the chatbot")
async def chat_endpoint(request: Request, body: ChatRequest):
    """
    Process a user message through the LangGraph conversational RAG pipeline.
    """
    start = time.perf_counter()

    session_id = body.session_id or f"sess_{uuid.uuid4().hex[:12]}"

    logger.info(
        "POST /api/chat | session=%s | msg='%s...'",
        session_id,
        body.message[:60],
    )

    if len(body.message) > settings.MAX_INPUT_LENGTH:
        elapsed = time.perf_counter() - start
        log_conversation(
            session_id=session_id,
            user_message=body.message[:200] + "... [TRUNCATED]",
            status="422 Unprocessable Content",
            duration=elapsed,
            error=f"Message exceeds maximum length of {settings.MAX_INPUT_LENGTH} characters.",
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Message exceeds maximum length of {settings.MAX_INPUT_LENGTH} characters.",
        )

    try:
        result = await run_chat(session_id=session_id, question=body.message)
    except Exception as exc:
        elapsed = time.perf_counter() - start
        logger.error("Unhandled error in chat pipeline: %s", exc, exc_info=True)
        log_conversation(
            session_id=session_id,
            user_message=body.message,
            status="500 Internal Server Error",
            duration=elapsed,
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred. Please try again.",
        )

    elapsed = time.perf_counter() - start

    # Logging per request: detected intent, rewritten query, number of docs retrieved, and which node answered
    logger.info(
        "[chat_endpoint] session=%s | intent=%s | rewritten_query=%s | docs_retrieved=%d | answering_node=%s | duration=%.2fs",
        session_id,
        result.get("intent"),
        result.get("rewritten_query"),
        len(result.get("sources", [])),
        result.get("answering_node"),
        elapsed,
    )

    sources = [
        SourceDoc(
            source=s.get("source", ""),
            page=s.get("page"),
            chunk_index=s.get("chunk_index"),
            title=s.get("title"),
            category=s.get("category"),
            as_of=s.get("as_of"),
        )
        for s in result.get("sources", [])
    ]

    # Clean dedicated conversation logging
    log_conversation(
        session_id=session_id,
        user_message=body.message,
        status="200 OK",
        duration=elapsed,
        sources=result.get("sources", []),
        llm_response=result.get("answer", ""),
    )

    return ChatResponse(
        session_id=session_id,
        answer=result.get("answer", ""),
        sources=sources,
        rewritten_query=result.get("rewritten_query"),
        intent=result.get("intent"),
        answering_node=result.get("answering_node"),
    )


@router.get("/health", response_model=HealthResponse, summary="Service health check")
async def health_check():
    """
    Check connectivity of Redis, ChromaDB, and LLM configuration.
    Does NOT expose credentials.
    """
    # Redis check
    redis_ok = await redis_manager.ping()
    redis_status = "connected" if redis_ok else "unavailable"

    # ChromaDB check (V2)
    try:
        chroma_svc = get_chroma_service_v2()
        count = chroma_svc.get_collection_count()
        chroma_status = f"connected ({count} docs)" if count >= 0 else "unavailable"
    except Exception:
        chroma_status = "unavailable"

    # LLM check
    try:
        llm_ok = bool(settings.OPENAI_API_KEY or settings.AZURE_OPENAI_API_KEY)
        llm_status = "configured" if llm_ok else "not configured"
    except Exception:
        llm_status = "unknown"

    overall = "healthy" if (redis_ok and "connected" in chroma_status and llm_ok) else "degraded"

    return HealthResponse(
        status=overall,
        redis=redis_status,
        chromadb=chroma_status,
        llm=llm_status,
        version=settings.APP_VERSION,
    )
