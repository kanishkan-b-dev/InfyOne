"""
Semantic retriever over ChromaDB.
Wraps the LangChain retriever with similarity threshold filtering and context length limits.
"""
import logging
from typing import Optional

from langchain_core.documents import Document

from app.core.config import get_settings
from app.rag.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)
_settings = get_settings()


def get_retriever(k: Optional[int] = None):
    """
    Return a LangChain retriever with similarity-score threshold.
    """
    vectorstore = get_vectorstore()
    retrieval_k = k or _settings.KNOWLEDGE_SEARCH_TOP_K

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": retrieval_k},
    )


def retrieve_documents(query: str, k: Optional[int] = None) -> list[Document]:
    """
    Perform semantic search and return relevant Document chunks.

    Documents are filtered by the configured similarity threshold.
    Returns an empty list when the collection is empty or query fails.
    """
    vectorstore = get_vectorstore()
    retrieval_k = k or _settings.KNOWLEDGE_SEARCH_TOP_K
    similarity_threshold = _settings.KNOWLEDGE_SIMILARITY_THRESHOLD

    search_query = query.strip()
    if len(search_query.split()) <= 2 and "athena" not in search_query.lower():
        expanded_query = f"{search_query} Athena Institute of Technology"
    else:
        expanded_query = search_query

    try:
        # Fetch candidate pool for hybrid re-ranking
        fetch_k = max(retrieval_k * 3, 15)
        results = vectorstore.similarity_search_with_relevance_scores(
            search_query, k=fetch_k
        )

        # For very short queries, check if contextual query yields better relevance
        if results and results[0][1] < similarity_threshold and expanded_query != search_query:
            expanded_results = vectorstore.similarity_search_with_relevance_scores(
                expanded_query, k=fetch_k
            )
            if expanded_results and expanded_results[0][1] > results[0][1]:
                results = expanded_results

        # Hybrid keyword re-ranking: boost candidates whose titles or content match query terms
        import re
        stop_words = {
            "what", "when", "where", "which", "who", "whom", "whose", "why",
            "how", "the", "and", "for", "ait", "athena", "institute", "technology",
            "does", "can", "are", "about", "with", "from", "that", "this"
        }
        query_terms = [
            t.lower() for t in re.findall(r"\w+", search_query)
            if len(t) > 2 and t.lower() not in stop_words
        ]

        scored_candidates = []
        for doc, sim_score in results:
            title = (doc.metadata.get("title") or doc.metadata.get("source") or "").lower()
            content = doc.page_content.lower()
            boost = 0.0
            for term in query_terms:
                if term in title:
                    boost += 0.20
                elif term in content:
                    boost += 0.05
            final_score = sim_score + boost
            scored_candidates.append((doc, final_score, sim_score))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        logger.info(
            "Retrieval: query='%s...', candidates=%d", query[:60], len(scored_candidates)
        )

        filtered: list[Document] = []
        for doc, boosted_score, sim_score in scored_candidates[:retrieval_k]:
            source_name = doc.metadata.get("title") or doc.metadata.get("source", "?")
            logger.debug("  score=%.4f (sim=%.4f) | source=%s", boosted_score, sim_score, source_name)
            if sim_score >= similarity_threshold or boosted_score >= similarity_threshold:
                filtered.append(doc)

        # Safety fallback: if no documents passed primary threshold,
        # but top candidate has reasonable semantic relevance (>= 0.15),
        # keep candidates >= 0.15 so valid short queries aren't falsely rejected
        if not filtered and scored_candidates and scored_candidates[0][2] >= 0.15:
            logger.info("Retrieval: applying fallback for top candidates with score >= 0.15")
            for doc, _, sim_score in scored_candidates[:retrieval_k]:
                if sim_score >= 0.15:
                    filtered.append(doc)

        logger.info(
            "Retrieval: %d/%d documents selected (threshold %.2f)",
            len(filtered),
            len(results),
            similarity_threshold,
        )
        return filtered

    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        return []


def format_context(documents: list[Document]) -> str:
    """
    Concatenate retrieved document chunks into a single context string.
    Ensures total context does not exceed KNOWLEDGE_MAX_CONTEXT_LENGTH.
    """
    if not documents:
        return ""
    parts = []
    max_len = _settings.KNOWLEDGE_MAX_CONTEXT_LENGTH

    total_len = 0
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("title") or doc.metadata.get("source") or "Athena Knowledge Base"
        category = doc.metadata.get("category")
        cat_str = f" [{category.upper()}]" if category else ""
        page = doc.metadata.get("page", "")
        page_str = f" (page {page})" if page else ""
        chunk_str = f"[Document {i} — {source}{cat_str}{page_str}]\n{doc.page_content}"

        if total_len + len(chunk_str) > max_len and parts:
            # Truncate remaining if exceeds max context length
            remaining = max_len - total_len
            if remaining > 100:
                parts.append(chunk_str[:remaining] + "...")
            break

        parts.append(chunk_str)
        total_len += len(chunk_str) + 6  # divider length

    return "\n\n---\n\n".join(parts)
