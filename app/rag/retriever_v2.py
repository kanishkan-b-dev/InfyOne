"""
V2 Retriever re-export for app.rag module hierarchy.
"""
from app.services.retriever_service import (
    RetrieverV2,
    RetrievedDocument,
    RetrievalResult,
    get_retriever_v2,
)

__all__ = [
    "RetrieverV2",
    "RetrievedDocument",
    "RetrievalResult",
    "get_retriever_v2",
]
