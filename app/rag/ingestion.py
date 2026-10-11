"""
Document ingestion pipeline for ChromaDB.
Called by scripts/ingest.py — NOT by the web server.
"""
import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.rag.vectorstore import get_chroma_client, get_vectorstore

logger = logging.getLogger(__name__)
_settings = get_settings()

SUPPORTED_EXTENSIONS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".md": TextLoader,   # Markdown is plain text — TextLoader works perfectly
}


def _doc_id(content: str, metadata: dict) -> str:
    """Stable deterministic ID for deduplication."""
    raw = f"{metadata.get('source','')}-{metadata.get('chunk_index','')}-{content[:200]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def load_document(file_path: Path) -> list[Document]:
    """Load a single document using the appropriate loader."""
    ext = file_path.suffix.lower()
    loader_cls = SUPPORTED_EXTENSIONS.get(ext)
    if not loader_cls:
        logger.warning("Skipping unsupported file: %s", file_path)
        return []

    try:
        # TextLoader needs explicit encoding to handle UTF-8 documents on Windows
        if loader_cls is TextLoader:
            loader = loader_cls(str(file_path), encoding="utf-8")
        else:
            loader = loader_cls(str(file_path))
        docs = loader.load()
        logger.info("Loaded %d pages from %s", len(docs), file_path.name)
        return docs
    except Exception as exc:
        logger.error("Failed to load %s: %s", file_path.name, exc)
        return []


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into meaningful chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_settings.CHUNK_SIZE,
        chunk_overlap=_settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split into %d chunks", len(chunks))
    return chunks


def enrich_metadata(chunks: list[Document], source_name: str) -> list[Document]:
    """Add/normalise metadata fields on each chunk."""
    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = source_name
        chunk.metadata["chunk_index"] = i
        # Normalise page field (PyPDF uses 'page', others may not)
        if "page" in chunk.metadata:
            chunk.metadata["page"] = int(chunk.metadata["page"]) + 1  # 1-indexed
    return chunks


def ingest_file(file_path: Path, force: bool = False) -> int:
    """
    Ingest a single file into ChromaDB.

    Returns the number of new chunks added.
    Skips chunks that already exist (deduplication via IDs).
    """
    raw_docs = load_document(file_path)
    if not raw_docs:
        return 0

    chunks = split_documents(raw_docs)
    chunks = enrich_metadata(chunks, file_path.name)

    vectorstore = get_vectorstore()

    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    ids = [_doc_id(t, m) for t, m in zip(texts, metadatas)]

    if not force:
        # Check which IDs already exist
        client = get_chroma_client()
        collection = client.get_or_create_collection(_settings.CHROMA_COLLECTION_NAME)
        existing = set(collection.get(ids=ids)["ids"])
        new_indices = [i for i, doc_id in enumerate(ids) if doc_id not in existing]

        if not new_indices:
            logger.info("All chunks already indexed for %s — skipping.", file_path.name)
            return 0

        texts = [texts[i] for i in new_indices]
        metadatas = [metadatas[i] for i in new_indices]
        ids = [ids[i] for i in new_indices]

    vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    logger.info("Added %d new chunks from %s", len(texts), file_path.name)
    return len(texts)


def ingest_directory(data_dir: str, force: bool = False) -> dict:
    """
    Ingest all supported documents in a directory recursively.

    Returns a summary dict: {filename: chunks_added}.
    """
    base = Path(data_dir)
    if not base.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    summary = {}
    files = [p for p in base.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not files:
        logger.warning("No supported documents found in %s", data_dir)
        return summary

    logger.info("Found %d documents to process in %s", len(files), data_dir)

    for file_path in sorted(files):
        count = ingest_file(file_path, force=force)
        summary[file_path.name] = count

    total = sum(summary.values())
    logger.info("Ingestion complete — total new chunks: %d", total)
    return summary
