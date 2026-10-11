"""
LangChain tools for the Athena Institute AI Chatbot.
Provides tools that the LLM orchestrator can call dynamically.
"""
import logging
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_core.tools import tool

from app.rag.retriever import format_context, retrieve_documents

logger = logging.getLogger(__name__)


@tool
def search_athena_knowledge_base(query: str) -> str:
    """
    Search the official Athena Institute of Technology knowledge base.

    Use this tool whenever the user asks about:
    - Undergraduate (B.Tech) & Postgraduate (M.Tech, MBA, PhD) programs and branches
    - Admission eligibility criteria, application process, and deadlines
    - Tuition fees, hostel fees, payment schedules, and scholarships
    - Campus amenities, library timings, research labs, sports facilities, hostel rooms
    - Placement statistics, average package, top recruiters, and career opportunities
    - Faculty members, departments, policies, and contact information

    Args:
        query: The semantic search query describing the information needed.

    Returns:
        Formatted excerpts from official institute documents.
    """
    docs = retrieve_documents(query)
    if not docs:
        return "No matching documents found in the Athena Institute knowledge base."
    return format_context(docs)


def execute_knowledge_search(query: str) -> Tuple[str, List[Document], List[dict]]:
    """
    Directly execute knowledge retrieval and return formatted context, documents, and source metadata.
    """
    docs = retrieve_documents(query)
    context = format_context(docs)
    sources = []
    for doc in docs:
        title = doc.metadata.get("title") or doc.metadata.get("source") or "Athena Knowledge Base"
        sources.append({
            "source": title,
            "title": title,
            "category": doc.metadata.get("category", ""),
            "page": doc.metadata.get("page"),
            "chunk_index": doc.metadata.get("chunk_index"),
        })
    return context, docs, sources
