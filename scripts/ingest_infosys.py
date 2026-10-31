#!/usr/bin/env python
"""
Ingest Infosys Knowledge Base into ChromaDB.

Reads 100 chunks from infosys_chroma_100_chunks.jsonl,
generates dense vector embeddings using OpenAI (via EmbeddingServiceV2),
and uploads (upserts) them with complete metadata into your ChromaDB collection.
"""
import json
import logging
import os
import sys
from pathlib import Path

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

# Load .env
env_path = ROOT_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

import chromadb
from app.services.embedding_service import get_embedding_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ingest_infosys")


def main():
    jsonl_path = ROOT_DIR / "infosys_chroma_100_chunks.jsonl"
    if not jsonl_path.exists():
        logger.error("Cannot find %s", jsonl_path)
        sys.exit(1)

    # 1. Read JSONL records
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    logger.info("Loaded %d records from %s", len(records), jsonl_path.name)

    # 2. Connect to ChromaDB
    api_key = os.getenv("CHROMA_CLOUD_API_KEY")
    tenant = os.getenv("CHROMA_CLOUD_TENANT") or os.getenv("CHROMA_TENANT") or "default_tenant"
    database = os.getenv("CHROMA_CLOUD_DATABASE") or os.getenv("CHROMA_DATABASE") or "default_database"
    cloud_host = os.getenv("CHROMA_CLOUD_HOST")
    collection_name = (
        os.getenv("CHROMA_COLLECTION_NAME")
        or os.getenv("CHROMA_COLLECTION_V2")
        or "InfosysCollection_v2"
    )
    metric = os.getenv("CHROMA_DISTANCE_METRIC", "cosine")

    logger.info("Target Chroma Collection: '%s' (distance metric: %s)", collection_name, metric)

    if api_key:
        logger.info("Connecting to Chroma Cloud (tenant=%s, database=%s)...", tenant, database)
        client = chromadb.CloudClient(
            api_key=api_key,
            tenant=tenant,
            database=database,
            cloud_host=cloud_host if cloud_host else "api.trychroma.com",
        )
    else:
        persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_data")
        logger.info("No CHROMA_CLOUD_API_KEY found, connecting to local Chroma: %s", persist_dir)
        client = chromadb.PersistentClient(path=persist_dir)

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": metric},
    )

    # 3. Generate Vector Embeddings
    logger.info("Initializing embedding service (OpenAI)...")
    embed_svc = get_embedding_service()
    logger.info("Embedding model: %s (dim=%d)", embed_svc.model_name, embed_svc.dimensions)

    doc_texts = [r["document"] for r in records]
    logger.info("Generating vector embeddings for %d document chunks...", len(doc_texts))
    embeddings = embed_svc.embed_documents(doc_texts)
    logger.info("Generated %d embedding vectors successfully.", len(embeddings))

    # 4. Prepare data for Chroma
    ids = []
    documents = []
    metadatas = []

    for r in records:
        ids.append(str(r["id"]))
        documents.append(str(r["document"]))
        meta = {
            "document_type": str(r.get("document_type", "knowledge")),
            "category": str(r.get("category", "general")),
            "chunk_index": int(r.get("chunk_index", 0)),
            "parent_doc_id": str(r.get("parent_doc_id", r["id"])),
            "total_chunks": int(r.get("total_chunks", 1)),
            "priority": int(r.get("priority", 0)),
            "tags": str(r.get("tags", "")),
            "created_at": str(r.get("created_at", "")),
            "title": str(r.get("title", "")),
            "source": str(r.get("source", "")),
            "as_of": str(r.get("as_of", "")),
        }
        metadatas.append(meta)

    # 5. Batch upsert into Chroma
    batch_size = 50
    logger.info("Upserting %d records into Chroma in batches of %d...", len(ids), batch_size)
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.upsert(
            ids=ids[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
            embeddings=embeddings[i:end],
        )
        logger.info("  Upserted batch %d - %d", i + 1, end)

    # 6. Verify count
    final_count = collection.count()
    print("\n" + "=" * 50)
    print("        INGESTION SUCCESSFUL")
    print("=" * 50)
    print(f"Collection Name:     {collection.name}")
    print(f"Total Chunks Stored: {final_count}")
    print(f"Embedding Dimension: {len(embeddings[0])}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()

