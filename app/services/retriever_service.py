"""
V2 Retrieval Service.
Implements multi-stage semantic retrieval using text-embedding-3-large over InfyOne / Infosys collection.

Architecture:
1. Candidate Generation: Vector search using Chroma cosine distance.
2. Metadata Filtering: Optional category / doc_type filtering.
3. Relevance Evaluation & Reranking: Term weighting, title/tag boosting, score normalization.
4. Candidate Selection: Filter by relevance threshold and select top context chunks.
5. Extensible: Designed for plugging in BM25 / Cross-Encoder rerankers.
"""
import os
import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.services.embedding_service import get_embedding_service, EmbeddingServiceV2
from app.services.chromadb_service import get_chroma_service_v2, ChromaServiceV2

logger = logging.getLogger(__name__)

STOP_WORDS = {
    "what", "when", "where", "which", "who", "whom", "whose", "why",
    "how", "the", "and", "for", "infyone", "infosys",
    "does", "can", "are", "about", "with", "from", "that", "this", "tell",
    "me", "is", "a", "an", "in", "to", "of", "do", "much"
}


@dataclass
class RetrievedDocument:
    id: str
    content: str
    metadata: Dict[str, Any]
    distance: float
    similarity: float
    rerank_score: float
    rank: int = 1

    @property
    def title(self) -> str:
        return self.metadata.get("title", "")

    @property
    def category(self) -> str:
        return self.metadata.get("category", "")

    @property
    def document_type(self) -> str:
        return self.metadata.get("document_type", "")


@dataclass
class RetrievalResult:
    query: str
    candidates: List[RetrievedDocument]
    selected_documents: List[RetrievedDocument]
    formatted_context: str
    metadata_filter: Optional[Dict[str, Any]] = None


class RetrieverV2:
    """
    V2 Retrieval Engine using text-embedding-3-large.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingServiceV2] = None,
        chroma_service: Optional[ChromaServiceV2] = None,
        default_candidate_k: int = 15,
        default_final_k: int = 5,
        similarity_threshold: float = 0.25,
    ):
        self.embedding_service = embedding_service or get_embedding_service()
        self.chroma_service = chroma_service or get_chroma_service_v2()
        self.default_candidate_k = default_candidate_k
        self.default_final_k = default_final_k
        self.similarity_threshold = similarity_threshold

    def _extract_query_terms(self, query: str) -> List[str]:
        """Extract meaningful terms from query for keyword/title matching."""
        tokens = re.findall(r"\w+", query.lower())
        return [t for t in tokens if len(t) >= 2 and t not in STOP_WORDS]

    def retrieve(
        self,
        query: str,
        final_k: Optional[int] = None,
        candidate_k: Optional[int] = None,
        where_filter: Optional[Dict[str, Any]] = None,
        min_threshold: Optional[float] = None,
    ) -> RetrievalResult:
        """
        Execute multi-stage retrieval:
        1. Embed query with text-embedding-3-large
        2. Retrieve candidate pool from Chroma V2
        3. Evaluate & rerank candidates
        4. Select top candidates passing threshold
        5. Format context
        """
        k_cand = candidate_k or self.default_candidate_k
        k_final = final_k or self.default_final_k
        threshold = min_threshold if min_threshold is not None else self.similarity_threshold

        clean_query = query.strip()
        if not clean_query:
            return RetrievalResult(
                query=query,
                candidates=[],
                selected_documents=[],
                formatted_context="",
            )

        # 1. Embed query
        query_vector = self.embedding_service.embed_text(clean_query)

        # 2. Vector search in Chroma V2 collection
        collection = self.chroma_service.get_collection()
        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": min(k_cand, collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        results = collection.query(**query_kwargs)

        doc_ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        # 3. Construct candidate pool & compute initial similarity (cosine similarity = 1 - distance)
        candidates: List[RetrievedDocument] = []
        query_terms = self._extract_query_terms(clean_query)

        for doc_id, text, meta, dist in zip(doc_ids, documents, metadatas, distances):
            # For cosine distance, distance in [0, 2].
            # Cosine similarity s = 1.0 - dist
            similarity = max(0.0, 1.0 - float(dist))

            # Reranking logic: Title match + Tag match + Term density
            boost = 0.0
            title_lower = (meta.get("title") or "").lower()
            tags_lower = (meta.get("tags") or "").lower()
            text_lower = text.lower()

            for term in query_terms:
                if term in title_lower:
                    boost += 0.15
                if term in tags_lower:
                    boost += 0.10
                elif term in text_lower:
                    boost += 0.04

            # Combined rerank score
            rerank_score = similarity + boost

            candidates.append(
                RetrievedDocument(
                    id=doc_id,
                    content=text,
                    metadata=meta or {},
                    distance=float(dist),
                    similarity=similarity,
                    rerank_score=rerank_score,
                )
            )

        # Sort candidates by rerank score descending
        candidates.sort(key=lambda x: x.rerank_score, reverse=True)

        for rank_idx, cand in enumerate(candidates, 1):
            cand.rank = rank_idx

        # 4. Filter by relevance threshold and select top k_final
        selected: List[RetrievedDocument] = []
        for cand in candidates[:k_final]:
            if cand.similarity >= threshold or cand.rerank_score >= threshold:
                selected.append(cand)

        # Fallback if top candidate has reasonable semantic relevance >= 0.20
        if not selected and candidates and candidates[0].similarity >= 0.20:
            selected = [candidates[0]]

        # 5. Format context string
        formatted_context = self.format_context(selected)

        return RetrievalResult(
            query=clean_query,
            candidates=candidates,
            selected_documents=selected,
            formatted_context=formatted_context,
            metadata_filter=where_filter,
        )

    def format_context(self, documents: List[RetrievedDocument]) -> str:
        """Format selected documents into grounded context block for LLM."""
        if not documents:
            return ""

        context_blocks = []
        for i, doc in enumerate(documents, 1):
            title = doc.title or "Document"
            cat = f" [{doc.category.upper()}]" if doc.category else ""
            block = f"[Context Document {i} — {title}{cat}]\n{doc.content}"
            context_blocks.append(block)

        return "\n\n---\n\n".join(context_blocks)


def get_retriever_v2() -> RetrieverV2:
    """Return a new or configured RetrieverV2 instance."""
    return RetrieverV2()
