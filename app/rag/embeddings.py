"""
Embeddings helper — thin wrapper around the provider factory.
"""
from app.llm.provider import get_embeddings  # re-export for convenience

__all__ = ["get_embeddings"]
