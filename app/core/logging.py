"""
Structured logging setup for the InfyOne AI Chatbot backend.
Supports console & file logging with JSON or standard text formatting based on .env.
"""
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys
from app.core.config import get_settings

settings = get_settings()


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": self.formatTime(record, self.datefmt or "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def setup_logging() -> None:
    """Configure application-wide logging based on .env settings."""
    level_name = getattr(settings, "LOG_LEVEL", "DEBUG" if settings.APP_DEBUG else "INFO").upper()
    log_level = getattr(logging, level_name, logging.INFO)

    is_json = getattr(settings, "LOG_FORMAT", "text").lower() == "json"

    if is_json:
        formatter = JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%SZ")
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # File Handler (if LOG_FILE_PATH is configured - terminal/system log)
    log_file_path = getattr(settings, "LOG_FILE_PATH", None)
    if log_file_path:
        try:
            log_path = Path(log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                str(log_path),
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            root_logger.addHandler(file_handler)
        except Exception as exc:
            print(f"Warning: Failed to setup log file at {log_file_path}: {exc}", file=sys.stderr)

    # Dedicated Conversation Logger setup
    setup_conversation_logger()

    # Quiet noisy libraries
    for noisy in ("httpx", "httpcore", "chromadb.telemetry", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def setup_conversation_logger() -> logging.Logger:
    """
    Configure a clean, isolated logger dedicated to chat exchanges (conversation.log).
    Free of HTTP request/response noise, debug dumps, and telemetry.
    """
    conv_logger = logging.getLogger("conversation")
    conv_logger.setLevel(logging.INFO)
    conv_logger.propagate = False  # Keep separate from terminal/app.log
    conv_logger.handlers.clear()

    conv_log_path = getattr(settings, "CONVERSATION_LOG_FILE_PATH", "logs/conversation.log")
    if conv_log_path:
        try:
            log_path = Path(conv_log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(
                str(log_path),
                maxBytes=20 * 1024 * 1024,  # 20MB
                backupCount=5,
                encoding="utf-8",
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            conv_logger.addHandler(handler)
        except Exception as exc:
            print(f"Warning: Failed to setup conversation log file at {conv_log_path}: {exc}", file=sys.stderr)
    return conv_logger


def log_conversation(
    session_id: str,
    user_message: str,
    status: str,
    duration: float,
    sources: list = None,
    llm_response: str = None,
    error: str = None,
) -> None:
    """
    Log a clear, human-readable record of a conversation exchange to conversation.log.
    Free of internal debug noise or JSON escaping.
    """
    from datetime import datetime
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sources = sources or []

    lines = [
        "=" * 80,
        f"TIMESTAMP    : {now_str}",
        f"SESSION ID   : {session_id}",
        f"STATUS       : {status} (duration: {duration:.2f}s)",
        f"USER MESSAGE : {user_message}",
    ]

    if error:
        lines.append(f"ERROR        : {error}")
    else:
        if sources:
            lines.append(f"RETRIEVED    : {len(sources)} source document(s)")
            for i, s in enumerate(sources, 1):
                src_name = s.get("source") or s.get("title") or "Unknown"
                cat = s.get("category")
                cat_str = f" [{cat.upper()}]" if cat else ""
                lines.append(f"  {i}. {src_name}{cat_str}")
        else:
            lines.append("RETRIEVED    : 0 documents (Direct conversational response / no knowledge chunks retrieved)")

        lines.append("LLM RESPONSE :")
        lines.append(llm_response or "(No response generated)")

    lines.append("=" * 80)
    lines.append("")  # Blank line separator between conversation entries

    entry = "\n".join(lines)
    conv_logger = logging.getLogger("conversation")
    # If handler hasn't been set up yet (e.g. before setup_logging called), set it up
    if not conv_logger.handlers:
        setup_conversation_logger()
    conv_logger.info(entry)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)
