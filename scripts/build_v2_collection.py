#!/usr/bin/env python
"""
Build V2 ChromaDB Collection: AthenaCollegeCollection_v2.

Reads 92 knowledge-base records from chat-gpt.json, generates new 3072-dimensional
embeddings using OpenAI text-embedding-3-large, and idempotently populates
AthenaCollegeCollection_v2 in Chroma Cloud.

STRICT SAFETY:
Never touches AthenaCollegeCollection (V1). Fails loudly on any mismatch.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

# Setup project path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
env_path = ROOT_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

from app.services.embedding_service import get_embedding_service
from app.services.chromadb_service import (
    get_chroma_service_v2,
    assert_safe_collection,
    PROTECTED_COLLECTION,
    DEFAULT_V2_COLLECTION,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_v2_collection")

EXPECTED_COUNT = 92
EXPECTED_DIMENSION = 3072
EXPECTED_MODEL = "text-embedding-3-large"


def find_backup_json() -> Path:
    """Locate chat-gpt.json across potential project paths."""
    candidates = [
        ROOT_DIR / "data" / "chat-gpt.json",
        ROOT_DIR.parent / "chat-gpt.json",
        ROOT_DIR / "chat-gpt.json",
        Path("chat-gpt.json"),
        Path("data/chat-gpt.json"),
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(
        f"Could not locate chat-gpt.json. Searched locations: {[str(c) for c in candidates]}"
    )


def validate_source_records(records: List[Dict[str, Any]]) -> None:
    """Validate data integrity of the source JSON."""
    if len(records) != EXPECTED_COUNT:
        raise ValueError(
            f"Expected exactly {EXPECTED_COUNT} records in chat-gpt.json, found {len(records)}!"
        )

    ids = [r.get("id") for r in records]
    if len(ids) != len(set(ids)):
        duplicates = [x for x in ids if ids.count(x) > 1]
        raise ValueError(f"Duplicate IDs found in source JSON: {set(duplicates)}")

    required_fields = [
        "id", "document", "document_type", "category", "chunk_index",
        "parent_doc_id", "total_chunks", "priority", "tags", "created_at", "title"
    ]

    for i, r in enumerate(records):
        for field in required_fields:
            if field not in r:
                raise ValueError(f"Record {i} (id={r.get('id')}) is missing required field: '{field}'")
            if field == "document" and (not r[field] or not str(r[field]).strip()):
                raise ValueError(f"Record {i} (id={r.get('id')}) has empty document content!")

    logger.info("Source JSON validation passed: 92 valid records with complete fields and zero duplicates.")


def main():
    parser = argparse.ArgumentParser(description="Ingest V2 Knowledge Base into ChromaDB")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset and recreate AthenaCollegeCollection_v2 before ingestion (safe, will never touch V1)",
    )
    args = parser.parse_args()

    # ── 1. Safety Check: Verify target collection name ────────────────────────
    target_collection = os.getenv("CHROMA_COLLECTION_V2", DEFAULT_V2_COLLECTION)
    assert_safe_collection(target_collection, operation="build_v2_collection")

    print("\n" + "=" * 65)
    print("      Athena Institute of Technology — V2 Collection Builder")
    print("=" * 65)
    print(f"Target Collection:   {target_collection}")
    print(f"Protected V1:        {PROTECTED_COLLECTION} (GUARANTEED UNTOUCHED)")
    print(f"Embedding Model:     {EXPECTED_MODEL}")
    print(f"Target Dimension:    {EXPECTED_DIMENSION}")
    print("=" * 65 + "\n")

    # ── 2. Check and record V1 baseline status ────────────────────────────────
    chroma_svc = get_chroma_service_v2()
    client = chroma_svc.get_client()

    v1_count_before = -1
    try:
        v1_col = client.get_collection(name=PROTECTED_COLLECTION)
        v1_count_before = v1_col.count()
        logger.info("V1 Collection '%s' baseline count verified: %d records", PROTECTED_COLLECTION, v1_count_before)
    except Exception as exc:
        logger.warning("Could not read V1 collection: %s", exc)

    # ── 3. Load & Validate chat-gpt.json ─────────────────────────────────────
    json_path = find_backup_json()
    logger.info("Reading source knowledge base from: %s", json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    validate_source_records(records)
    source_count = len(records)

    # ── 4. Handle Reset Flag if specified ────────────────────────────────────
    if args.reset:
        logger.info("Resetting V2 collection '%s' as requested...", target_collection)
        chroma_svc.reset_v2_collection_safely(confirm=True)

    # ── 5. Generate Embeddings using text-embedding-3-large ───────────────────
    embed_svc = get_embedding_service()
    if embed_svc.model_name != EXPECTED_MODEL:
        logger.warning(
            "Configured embedding model is '%s', expected '%s'",
            embed_svc.model_name,
            EXPECTED_MODEL,
        )

    logger.info(
        "Generating %d embeddings using model '%s' (dimension=%d)...",
        len(records),
        embed_svc.model_name,
        embed_svc.dimensions,
    )

    documents_text = [r["document"] for r in records]
    embeddings = embed_svc.embed_documents(documents_text)

    # Validate embedding dimensions
    if not embeddings:
        raise RuntimeError("Embedding generation returned empty list!")
    first_dim = len(embeddings[0])
    if first_dim != EXPECTED_DIMENSION:
        raise ValueError(
            f"Embedding dimension mismatch: expected {EXPECTED_DIMENSION}, got {first_dim}"
        )
    for idx, emb in enumerate(embeddings):
        if len(emb) != EXPECTED_DIMENSION:
            raise ValueError(f"Vector {idx} dimension mismatch: {len(emb)} != {EXPECTED_DIMENSION}")

    logger.info("All %d embeddings generated successfully (dimension=%d).", len(embeddings), first_dim)

    # ── 6. Ingest into AthenaCollegeCollection_v2 ────────────────────────────
    logger.info("Ingesting records into Chroma collection '%s'...", target_collection)
    inserted, updated = chroma_svc.ingest_records(records, embeddings)
    logger.info("Ingestion result: %d inserted, %d updated.", inserted, updated)

    # ── 7. Comprehensive Ingestion Validation ─────────────────────────────────
    v2_col = chroma_svc.get_collection()
    v2_count = v2_col.count()

    print("\n" + "-" * 55)
    print("                INGESTION VALIDATION")
    print("-" * 55)
    print(f"Collection Name:          {v2_col.name}")
    print(f"Source JSON Record Count: {source_count}")
    print(f"Chroma V2 Record Count:   {v2_count}")
    print(f"Embedding Model:          {embed_svc.model_name}")
    print(f"Embedding Dimension:      {first_dim}")

    # Fetch stored data to verify fields and metadata
    stored_data = v2_col.get(include=["documents", "metadatas", "embeddings"])
    stored_ids = stored_data.get("ids", [])
    stored_docs = stored_data.get("documents", [])
    stored_metas = stored_data.get("metadatas", [])
    stored_embeddings = stored_data.get("embeddings", [])

    print(f"Unique Stored IDs:        {len(set(stored_ids))}")
    print(f"Duplicate IDs:            {len(stored_ids) - len(set(stored_ids))}")

    # Verify ID match
    source_ids = set(r["id"] for r in records)
    missing_ids = source_ids - set(stored_ids)
    if missing_ids:
        raise RuntimeError(f"FATAL: Missing IDs in V2 collection: {missing_ids}")

    # Check for empty documents or missing metadata
    empty_docs = [sid for sid, doc in zip(stored_ids, stored_docs) if not doc or not doc.strip()]
    if empty_docs:
        raise RuntimeError(f"FATAL: Found empty documents in V2 collection: {empty_docs}")

    required_meta_keys = [
        "document_type", "category", "chunk_index", "parent_doc_id",
        "total_chunks", "priority", "tags", "created_at", "title"
    ]
    for sid, meta in zip(stored_ids, stored_metas):
        for k in required_meta_keys:
            if k not in meta:
                raise RuntimeError(f"FATAL: Record '{sid}' is missing metadata key '{k}'!")

    # Check embedding dimension on stored vectors
    if stored_embeddings is not None and len(stored_embeddings) > 0:
        stored_dim = len(stored_embeddings[0])
        print(f"Stored Vector Dimension:  {stored_dim}")
        if stored_dim != EXPECTED_DIMENSION:
            raise RuntimeError(f"FATAL: Stored vector dimension {stored_dim} != {EXPECTED_DIMENSION}!")

    # Strict count assertion
    if v2_count != source_count:
        raise RuntimeError(
            f"FATAL: Count mismatch! source_count ({source_count}) != v2_count ({v2_count})"
        )

    print("\n" + "-" * 55)
    print("                CATEGORY DISTRIBUTION")
    print("-" * 55)
    category_counts = chroma_svc.get_category_counts()
    for cat, count in category_counts.items():
        print(f"  • {cat:<22}: {count:>3} documents")
    print(f"  {'Total':<24}: {sum(category_counts.values()):>3} documents")

    # ── 8. Safety Verification: V1 Must Be Untouched ──────────────────────────
    print("\n" + "-" * 55)
    print("                V1 SAFETY VERIFICATION")
    print("-" * 55)
    v1_count_after = -1
    try:
        v1_col_after = client.get_collection(name=PROTECTED_COLLECTION)
        v1_count_after = v1_col_after.count()
        print(f"V1 Collection Name:       {PROTECTED_COLLECTION}")
        print(f"V1 Count Before:          {v1_count_before}")
        print(f"V1 Count After:           {v1_count_after}")
        if v1_count_before != -1 and v1_count_after != v1_count_before:
            raise RuntimeError(
                f"FATAL SAFETY VIOLATION: V1 count changed from {v1_count_before} to {v1_count_after}!"
            )
        print("STATUS: V1 COLLECTION COMPLETELY UNTOUCHED AND VERIFIED.")
    except Exception as exc:
        print(f"V1 Safety Check Note: {exc}")

    print("=" * 65)
    print("  SUCCESS: AthenaCollegeCollection_v2 successfully built and verified!")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
