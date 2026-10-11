"""
ChromaDB vector store wrapper.

Supports Chroma Cloud, remote Chroma HTTP server, and local persistent ChromaDB.
Credentials and settings are read from .env via app.core.config.Settings.
"""
import logging
import os
from functools import lru_cache
from urllib.parse import urlparse

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma

from app.core.config import get_settings
from app.llm.provider import get_embeddings

logger = logging.getLogger(__name__)
_settings = get_settings()

# Suppress ChromaDB telemetry
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")


@lru_cache()
def get_chroma_client() -> chromadb.api.ClientAPI:
    """
    Build and cache a ChromaDB client based on .env configuration.

    1. Chroma Cloud / Remote: If CHROMA_CLOUD_API_KEY, CHROMA_CLOUD_HOST, or CHROMA_CLOUD_TENANT is provided.
    2. Local PersistentClient: Fallback to local persistence directory.
    """
    api_key = _settings.CHROMA_CLOUD_API_KEY
    tenant = _settings.CHROMA_CLOUD_TENANT or "default_tenant"
    database = _settings.CHROMA_CLOUD_DATABASE or "default_database"
    cloud_host_url = _settings.CHROMA_CLOUD_HOST

    # Chroma Cloud connection
    if api_key or cloud_host_url or _settings.CHROMA_CLOUD_TENANT:
        if cloud_host_url:
            parsed = urlparse(cloud_host_url)
            host = parsed.hostname or cloud_host_url
            port = parsed.port or (443 if parsed.scheme == "https" else 8000)
            ssl = parsed.scheme == "https"

            # Check if this is Chroma Cloud (CloudClient) or standard HttpClient
            if "trychroma.com" in host or (api_key and tenant and database):
                try:
                    client = chromadb.CloudClient(
                        tenant=tenant,
                        database=database,
                        api_key=api_key or "",
                        cloud_host=host,
                        cloud_port=port,
                        enable_ssl=ssl,
                    )
                    logger.info(
                        "ChromaDB CloudClient connected — host=%s port=%d tenant=%s database=%s",
                        host, port, tenant, database,
                    )
                    return client
                except Exception as exc:
                    logger.warning("CloudClient initialization note: %s, trying HttpClient...", exc)

            headers = {}
            if api_key:
                headers["x-chroma-token"] = api_key
                headers["Authorization"] = f"Bearer {api_key}"

            client = chromadb.HttpClient(
                host=host,
                port=port,
                ssl=ssl,
                headers=headers if headers else None,
                tenant=tenant,
                database=database,
            )
            logger.info(
                "ChromaDB HttpClient connected — host=%s port=%d ssl=%s tenant=%s database=%s",
                host, port, ssl, tenant, database,
            )
            return client

        elif api_key and (tenant or database):
            try:
                client = chromadb.CloudClient(
                    tenant=tenant,
                    database=database,
                    api_key=api_key,
                )
                logger.info(
                    "ChromaDB CloudClient connected — tenant=%s database=%s",
                    tenant, database,
                )
                return client
            except Exception as exc:
                if "does not match" in str(exc):
                    import re
                    m = re.search(r"does not match ([0-9a-fA-F-]+) from the server", str(exc))
                    if m:
                        server_tenant = m.group(1)
                        logger.warning(
                            "Chroma Cloud tenant mismatch: switching from '%s' to server tenant '%s'",
                            tenant, server_tenant,
                        )
                        client = chromadb.CloudClient(
                            tenant=server_tenant,
                            database=database,
                            api_key=api_key,
                        )
                        return client
                raise

    # Local fallback
    client = chromadb.PersistentClient(
        path=_settings.CHROMA_PERSIST_DIRECTORY,
    )
    logger.info(
        "ChromaDB [local] initialized — path=%s",
        _settings.CHROMA_PERSIST_DIRECTORY,
    )
    return client


@lru_cache()
def get_vectorstore() -> Chroma:
    """Return a cached LangChain Chroma vector store backed by the configured client."""
    embeddings = get_embeddings()
    collection_metadata = {}
    if _settings.CHROMA_DISTANCE_METRIC:
        collection_metadata["hnsw:space"] = _settings.CHROMA_DISTANCE_METRIC

    vectorstore = Chroma(
        client=get_chroma_client(),
        collection_name=_settings.CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        collection_metadata=collection_metadata if collection_metadata else None,
    )
    logger.info(
        "Chroma vector store ready — collection=%s distance_metric=%s",
        _settings.CHROMA_COLLECTION_NAME,
        _settings.CHROMA_DISTANCE_METRIC,
    )
    return vectorstore


def get_collection_count() -> int:
    """Return the number of vectors in the collection (used by /health endpoint)."""
    try:
        client = get_chroma_client()
        collection = client.get_or_create_collection(
            name=_settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": _settings.CHROMA_DISTANCE_METRIC} if _settings.CHROMA_DISTANCE_METRIC else None,
        )
        return collection.count()
    except Exception as exc:
        logger.warning("ChromaDB count failed: %s", exc)
        return -1
