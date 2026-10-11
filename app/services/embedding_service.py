"""
V2 Embedding Service.
Uses OpenAI text-embedding-3-large for high-fidelity semantic representations.
Reads configuration from environment variables (.env).
"""
import os
import logging
from typing import List, Optional
from functools import lru_cache

from openai import OpenAI
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


class EmbeddingServiceV2:
    """
    Manages embedding generation for the V2 architecture using text-embedding-3-large.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        dimensions: Optional[int] = None,
        api_key: Optional[str] = None,
        batch_size: int = 100,
    ):
        self.model_name = (
            model_name
            or os.getenv("EMBEDDING_MODEL_V2")
            or os.getenv("EMBEDDING_MODEL")
            or "text-embedding-3-large"
        )
        dim_env = os.getenv("EMBEDDING_DIMENSION_V2")
        self.dimensions = dimensions or (int(dim_env) if dim_env else 3072)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.batch_size = batch_size

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Please set it in your environment or .env file."
            )

        self._client = OpenAI(api_key=self.api_key)
        logger.info(
            "EmbeddingServiceV2 initialized with model=%s, dimensions=%d",
            self.model_name,
            self.dimensions,
        )

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single string."""
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")

        response = self._client.embeddings.create(
            input=[text],
            model=self.model_name,
            dimensions=self.dimensions,
        )
        return response.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for a list of document strings in batches.
        Embeddings are generated only for document text.
        """
        if not texts:
            return []

        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self._client.embeddings.create(
                input=batch,
                model=self.model_name,
                dimensions=self.dimensions,
            )
            batch_vectors = [item.embedding for item in response.data]
            all_embeddings.extend(batch_vectors)

        return all_embeddings

    def get_langchain_embeddings(self) -> OpenAIEmbeddings:
        """Return a LangChain-compatible OpenAIEmbeddings instance."""
        return OpenAIEmbeddings(
            api_key=self.api_key,
            model=self.model_name,
            dimensions=self.dimensions,
            chunk_size=self.batch_size,
        )


@lru_cache()
def get_embedding_service() -> EmbeddingServiceV2:
    """Return a cached singleton instance of EmbeddingServiceV2."""
    return EmbeddingServiceV2()
