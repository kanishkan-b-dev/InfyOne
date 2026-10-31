"""
FastAPI application entry point for the InfyOne / Infosys AI Chatbot.

All configuration is sourced from .env via app.core.config.Settings.
APP_HOST, APP_PORT, APP_ENV, APP_CORS_ORIGINS, APP_ALLOWED_HOSTS, etc. are read from .env.
"""
import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.chat import router as chat_router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.memory.redis_manager import redis_manager

setup_logging()
logger = logging.getLogger(__name__)
settings = get_settings()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# Simple rate limiter tracking per IP
_rate_limit_records = defaultdict(list)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown lifecycle events."""
    # ── Startup ──────────────────────────────────────────────────
    logger.info(
        "Starting %s v%s [env=%s debug=%s]",
        settings.APP_NAME,
        settings.APP_VERSION,
        settings.APP_ENV,
        settings.APP_DEBUG,
    )
    logger.info(
        "ChromaDB V2 collection=%s metric=%s",
        settings.CHROMA_COLLECTION_V2,
        settings.CHROMA_DISTANCE_METRIC,
    )

    try:
        await redis_manager.connect()
    except Exception as exc:
        logger.warning("Redis unavailable at startup: %s — session memory will not work.", exc)

    # Pre-warm the V2 Chroma connection
    try:
        from app.services.chromadb_service import get_chroma_service_v2
        get_chroma_service_v2().get_collection()
    except Exception as exc:
        logger.warning("ChromaDB V2 pre-warm: %s", exc)

    logger.info(
        "Server ready at http://%s:%s",
        settings.APP_HOST,
        settings.APP_PORT,
    )
    yield

    # ── Shutdown ─────────────────────────────────────────────────
    await redis_manager.disconnect()
    logger.info("Application shutdown complete.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=f"RAG-powered AI chatbot for {settings.APP_NAME}",
    docs_url="/docs" if settings.APP_ENV == "development" else None,
    redoc_url=None,
    lifespan=lifespan,
)

# ── Rate Limiting Middleware ────────────────────────────────────
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if settings.RATE_LIMIT_PER_MINUTE > 0 and request.url.path.startswith("/api/"):
        client_ip = request.client.host if request.client else "127.0.0.1"
        now = time.time()
        # Clean timestamps older than 60s
        timestamps = [t for t in _rate_limit_records[client_ip] if now - t < 60]
        if len(timestamps) >= settings.RATE_LIMIT_PER_MINUTE:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded. Please wait a moment."},
            )
        timestamps.append(now)
        _rate_limit_records[client_ip] = timestamps

    response = await call_next(request)
    return response


# ── Allowed Hosts Middleware ────────────────────────────────────
if settings.allowed_hosts and "*" not in settings.allowed_hosts:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.allowed_hosts,
    )

# ── CORS ────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)

# ── API Routers ──────────────────────────────────────────────────
app.include_router(chat_router)

# ── Static Files & Frontend ─────────────────────────────────────
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/", include_in_schema=False)
async def root():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(
            str(index_file),
            headers={
                # Always fetch fresh HTML — prevents stale UI after redeploys
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "online",
        "docs_url": "/docs" if settings.APP_ENV == "development" else None,
    }

@app.get("/api/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV,
    }


# ── Direct run (python -m app.main) — reads APP_HOST/APP_PORT ────
if __name__ == "__main__":
    import os
    import uvicorn

    reload = settings.APP_ENV == "development"
    port = int(os.environ.get("PORT", settings.effective_port))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=reload,
        log_level=settings.LOG_LEVEL.lower() if hasattr(settings, "LOG_LEVEL") else "info",
    )
