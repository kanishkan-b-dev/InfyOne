"""
LLM and Embeddings provider factory.
Supports OpenAI and Azure OpenAI with full configuration via .env.
"""
from functools import lru_cache
import logging
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI, OpenAIEmbeddings

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@lru_cache()
def get_llm():
    """Return a cached LangChain Chat LLM instance."""
    if settings.use_azure:
        logger.info("LLM provider: Azure OpenAI (deployment=%s)", settings.AZURE_OPENAI_CHAT_DEPLOYMENT)
        return AzureChatOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            azure_deployment=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            timeout=settings.LLM_TIMEOUT,
        )

    if not settings.OPENAI_API_KEY:
        raise RuntimeError(
            "No LLM credentials found. Set OPENAI_API_KEY or AZURE_OPENAI_* variables in .env"
        )

    logger.info("LLM provider: OpenAI (model=%s)", settings.OPENAI_MODEL)
    is_reasoning = settings.OPENAI_MODEL.lower().startswith(("o1", "o3", "o4"))
    kwargs = {
        "api_key": settings.OPENAI_API_KEY,
        "model": settings.OPENAI_MODEL,
        "timeout": settings.LLM_TIMEOUT,
    }
    if is_reasoning:
        kwargs["max_completion_tokens"] = settings.LLM_MAX_TOKENS
    else:
        kwargs["temperature"] = settings.LLM_TEMPERATURE
        kwargs["max_tokens"] = settings.LLM_MAX_TOKENS

    return ChatOpenAI(**kwargs)


@lru_cache()
def get_embeddings():
    """Return a cached LangChain Embeddings instance based on .env configuration."""
    provider = settings.EMBEDDING_PROVIDER.lower() if settings.EMBEDDING_PROVIDER else "openai"

    if provider == "azure" or (settings.use_azure and provider != "openai"):
        logger.info(
            "Embeddings provider: Azure OpenAI (deployment=%s)",
            settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        )
        return AzureOpenAIEmbeddings(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            azure_deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )

    # Default to OpenAI
    logger.info(
        "Embeddings provider: OpenAI (model=%s dimension=%d batch_size=%d)",
        settings.EMBEDDING_MODEL,
        settings.EMBEDDING_DIMENSION,
        settings.EMBEDDING_BATCH_SIZE,
    )

    kwargs = {
        "api_key": settings.OPENAI_API_KEY,
        "model": settings.EMBEDDING_MODEL,
        "chunk_size": settings.EMBEDDING_BATCH_SIZE,
    }
    if "text-embedding-3" in settings.EMBEDDING_MODEL and settings.EMBEDDING_DIMENSION:
        kwargs["dimensions"] = settings.EMBEDDING_DIMENSION

    return OpenAIEmbeddings(**kwargs)
