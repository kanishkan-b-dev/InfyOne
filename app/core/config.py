"""
Central configuration system using Pydantic Settings.

ALL settings are read exclusively from environment variables / .env file.
Matches standard application and cloud service environment variables.
"""
import json
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Union
from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"
if _ENV_FILE.exists():
    load_dotenv(dotenv_path=_ENV_FILE, override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE if _ENV_FILE.exists() else None,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application Settings ─────────────────────────────────────────────────
    APP_NAME: str = "InfyOne AI Assistant"
    APP_ENV: str = "production"
    APP_DEBUG: bool = False
    APP_VERSION: str = "1.0.0"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    PORT: Optional[int] = None
    APP_SECRET_KEY: str = "0fc53de77df447e9dd462248fc90e50c48819fd73f0abd1c3883331cf3cedaf1"
    APP_ALLOWED_HOSTS: Union[str, List[str]] = ["*"]
    APP_CORS_ORIGINS: Union[str, List[str]] = ["*"]
    RATE_LIMIT_PER_MINUTE: int = 60

    # ── Redis Cloud Settings ─────────────────────────────────────────────────
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    REDIS_SSL: bool = False
    REDIS_SESSION_TTL: int = 3600   # seconds — 1 hour per session
    REDIS_MAX_HISTORY: int = 20     # max message pairs stored per session
    REDIS_URL: Optional[str] = None

    # ── Chroma Cloud Settings ────────────────────────────────────────────────
    CHROMA_CLOUD_HOST: Optional[str] = None
    CHROMA_CLOUD_API_KEY: Optional[str] = None
    CHROMA_CLOUD_TENANT: Optional[str] = None
    CHROMA_CLOUD_DATABASE: Optional[str] = None
    CHROMA_TENANT: Optional[str] = None
    CHROMA_DATABASE: Optional[str] = None
    CHROMA_COLLECTION_NAME: str = "InfyOne"
    CHROMA_COLLECTION_V2: str = "InfyOne"
    CHROMA_DISTANCE_METRIC: str = "cosine"
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_data"
    CHROMA_MODE: Optional[str] = None

    # ── Embedding Settings (V2 Default: text-embedding-3-large 3072-dim) ─────
    EMBEDDING_PROVIDER: str = "openai"
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    EMBEDDING_DIMENSION: int = 3072
    EMBEDDING_MODEL_V2: str = "text-embedding-3-large"
    EMBEDDING_DIMENSION_V2: int = 3072
    EMBEDDING_BATCH_SIZE: int = 100

    # ── OpenAI API Settings ──────────────────────────────────────────────────
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"

    # ── Azure OpenAI (optional) ──────────────────────────────────────────────
    AZURE_OPENAI_API_KEY: Optional[str] = None
    AZURE_OPENAI_ENDPOINT: Optional[str] = None
    AZURE_OPENAI_API_VERSION: str = "2024-02-01"
    AZURE_OPENAI_CHAT_DEPLOYMENT: Optional[str] = None
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: Optional[str] = None

    # ── Knowledge Retrieval Settings ─────────────────────────────────────────
    KNOWLEDGE_SEARCH_TOP_K: int = 5
    KNOWLEDGE_SIMILARITY_THRESHOLD: float = 0.25
    KNOWLEDGE_MAX_CONTEXT_LENGTH: int = 4000
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150

    # ── Logging Settings ─────────────────────────────────────────────────────
    LOG_LEVEL: str = "DEBUG"
    LOG_FORMAT: str = "json"
    LOG_FILE_PATH: str = "logs/app.log"
    CONVERSATION_LOG_FILE_PATH: str = "logs/conversation.log"

    # ── LLM & Execution Settings ─────────────────────────────────────────────
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 1024
    LLM_TIMEOUT: int = 60
    MAX_INPUT_LENGTH: int = 2000

    @field_validator("APP_ALLOWED_HOSTS", "APP_CORS_ORIGINS", mode="before")
    @classmethod
    def parse_json_or_list(cls, v):
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("[") and v.endswith("]"):
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [x.strip() for x in v.split(",") if x.strip()]
        return v

    # ── Derived & compatibility properties ───────────────────────────────────
    @property
    def cors_origins(self) -> List[str]:
        if isinstance(self.APP_CORS_ORIGINS, list):
            return self.APP_CORS_ORIGINS
        return ["*"]

    @property
    def allowed_hosts(self) -> List[str]:
        if isinstance(self.APP_ALLOWED_HOSTS, list):
            hosts = list(self.APP_ALLOWED_HOSTS)
            if "testserver" not in hosts and "*" not in hosts:
                hosts.append("testserver")
            return hosts
        return ["*"]

    @property
    def use_azure(self) -> bool:
        return bool(self.AZURE_OPENAI_API_KEY and self.AZURE_OPENAI_ENDPOINT)

    @property
    def DEBUG(self) -> bool:
        return self.APP_DEBUG

    @property
    def HOST(self) -> str:
        return self.APP_HOST

    @property
    def effective_port(self) -> int:
        return self.PORT if self.PORT is not None else self.APP_PORT

    @property
    def ALLOWED_ORIGINS(self) -> List[str]:
        return self.cors_origins

    @property
    def RETRIEVAL_K(self) -> int:
        return self.KNOWLEDGE_SEARCH_TOP_K

    @property
    def SIMILARITY_THRESHOLD(self) -> float:
        return self.KNOWLEDGE_SIMILARITY_THRESHOLD


@lru_cache()
def get_settings() -> Settings:
    """Return cached singleton Settings instance (loaded from .env)."""
    return Settings()
