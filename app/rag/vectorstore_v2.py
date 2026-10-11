"""
V2 Vector Store re-export for app.rag module hierarchy.
"""
from app.services.chromadb_service import (
    ChromaServiceV2,
    get_chroma_service_v2,
    assert_safe_collection,
    DEFAULT_V2_COLLECTION,
    PROTECTED_COLLECTION,
)

__all__ = [
    "ChromaServiceV2",
    "get_chroma_service_v2",
    "assert_safe_collection",
    "DEFAULT_V2_COLLECTION",
    "PROTECTED_COLLECTION",
]
