"""
LangChain tools for the Infosys AI Chatbot.
Exposes retrieval over the ChromaDB collection without hard category filtering,
using category as a soft ranking hint, deprioritizing FAQ chunks in favor of
authoritative non-FAQ knowledge, with automatic de-duplication and logging.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import tool

from app.services.chromadb_service import get_chroma_service_v2
from app.services.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)

VALID_CATEGORIES = {
    "overview", "leadership", "financials", "acquisitions",
    "services", "platforms", "ai", "deals", "operations",
    "people", "esg", "risk", "faq"
}


def retrieve_infosys_documents(
    query: str,
    category: Optional[str] = None,
    top_k: int = 8,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Execute semantic search over the Infosys Chroma collection.
    - Soft ranking boost for category matches.
    - FAQ chunks have source set to 'curated-faq' and are ranked lower so primary chunks are preferred.
    - De-duplicates chunks.
    """
    embed_svc = get_embedding_service()
    query_vector = embed_svc.embed_text(query)

    chroma_svc = get_chroma_service_v2()
    collection = chroma_svc.get_collection()

    pool_size = min(top_k + 8, 100)
    try:
        res = collection.query(query_embeddings=[query_vector], n_results=pool_size)
    except Exception as exc:
        logger.error("[retrieve] Collection query error: %s", exc)
        return [], ""

    documents: List[Dict[str, Any]] = []
    seen_ids = set()
    seen_keys = set()

    if res and res.get("ids") and len(res["ids"][0]) > 0:
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0] if res.get("metadatas") else [{}] * len(ids)
        distances = res.get("distances", [[0.0] * len(ids)])[0]

        scored_candidates = []
        target_cat = category.strip().lower() if category and category.strip().lower() in VALID_CATEGORIES else None

        for doc_id, text, meta, dist in zip(ids, docs, metas, distances):
            meta_dict = meta or {}
            title = meta_dict.get("title") or "Infosys Document"
            cat = str(meta_dict.get("category") or "general").lower()
            source = meta_dict.get("source") or "https://www.infosys.com"
            as_of = meta_dict.get("as_of") or "N/A"

            # Requirement: Change source metadata of FAQ chunks to 'curated-faq'
            is_faq = (cat == "faq")
            if is_faq:
                source = "curated-faq"

            effective_score = dist
            # Prefer non-FAQ chunks containing the same fact by slightly deprioritizing FAQ
            if is_faq:
                effective_score += 0.05
            elif target_cat and cat == target_cat:
                effective_score -= 0.05

            scored_candidates.append((effective_score, doc_id, text, title, cat, source, as_of))

        # Sort candidate pool
        scored_candidates.sort(key=lambda item: item[0])

        for _, doc_id, text, title, cat, source, as_of in scored_candidates:
            norm_title = title.strip().lower()
            norm_content_prefix = text.strip()[:80].lower()
            dedup_key = (norm_title, norm_content_prefix)

            if doc_id in seen_ids or dedup_key in seen_keys:
                continue

            seen_ids.add(doc_id)
            seen_keys.add(dedup_key)

            documents.append({
                "id": doc_id,
                "content": text,
                "metadata": {
                    "title": title,
                    "category": cat,
                    "source": source,
                    "as_of": as_of,
                },
            })

            if len(documents) >= top_k:
                break

    retrieved_titles = [d["metadata"].get("title") for d in documents]
    logger.info(
        "[retrieve] Filter used: %s | Retrieved %d titles: %s",
        category,
        len(documents),
        retrieved_titles,
    )

    context_blocks: List[str] = []
    for i, doc in enumerate(documents, 1):
        meta = doc.get("metadata", {})
        block = (
            f"[Document {i}]\n"
            f"Title: {meta.get('title')}\n"
            f"Category: {meta.get('category')}\n"
            f"Source: {meta.get('source')}\n"
            f"As of: {meta.get('as_of')}\n"
            f"Content: {doc.get('content')}"
        )
        context_blocks.append(block)

    formatted_context = "\n\n---\n\n".join(context_blocks)
    return documents, formatted_context


@tool
def search_infosys_knowledge_base(query: str, category: Optional[str] = None) -> str:
    """
    Search the official Infosys knowledge base for facts about Infosys.
    """
    _, context = retrieve_infosys_documents(query, category=category, top_k=8)
    if not context:
        return "No matching documents found in the Infosys knowledge base."
    return context


# Backwards compatibility alias
search_infyone_knowledge_base = search_infosys_knowledge_base
