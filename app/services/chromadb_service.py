"""
V2 ChromaDB Service.
Manages Chroma Cloud connection and operations for InfyOne / Infosys collection.

STRICT SAFETY POLICY:
The legacy collection is protected.
All destructive, clear, overwrite, or modification operations on legacy collections
are strictly blocked and will raise a PermissionError.
"""
import os
import logging
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

logger = logging.getLogger(__name__)

# ── Safety Configuration ─────────────────────────────────────────────────────
PROTECTED_COLLECTION = os.getenv("CHROMA_PROTECTED_COLLECTION", "legacy_protected_collection")
DEFAULT_V2_COLLECTION = "InfyOne"


def assert_safe_collection(collection_name: str, operation: str = "modify") -> None:
    """
    Enforce safety check preventing any destructive or mutating operation
    on the protected legacy production collection.
    """
    if collection_name and collection_name == PROTECTED_COLLECTION:
        raise PermissionError(
            f"CRITICAL SAFETY VIOLATION: Refusing to execute '{operation}' against protected "
            f"collection '{PROTECTED_COLLECTION}'! Legacy collection must remain completely untouched."
        )


class ChromaServiceV2:
    """
    ChromaDB Service dedicated to V2 architecture operations.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        distance_metric: str = "cosine",
    ):
        self.collection_name = (
            collection_name
            or os.getenv("CHROMA_COLLECTION_NAME")
            or os.getenv("CHROMA_COLLECTION_V2")
            or DEFAULT_V2_COLLECTION
        )
        self.distance_metric = (
            os.getenv("CHROMA_DISTANCE_METRIC") or distance_metric or "cosine"
        )
        self._client: Optional[ClientAPI] = None

    def get_client(self) -> ClientAPI:
        """Connect to Chroma Cloud or local instance based on environment variables."""
        if self._client is not None:
            return self._client

        api_key = os.getenv("CHROMA_CLOUD_API_KEY")
        tenant = (
            os.getenv("CHROMA_CLOUD_TENANT")
            or os.getenv("CHROMA_TENANT")
            or "default_tenant"
        )
        database = (
            os.getenv("CHROMA_CLOUD_DATABASE")
            or os.getenv("CHROMA_DATABASE")
            or "default_database"
        )
        cloud_host = os.getenv("CHROMA_CLOUD_HOST")

        if api_key:
            logger.info(
                "Connecting to Chroma Cloud (tenant=%s, database=%s)...",
                tenant,
                database,
            )
            self._client = chromadb.CloudClient(
                api_key=api_key,
                tenant=tenant,
                database=database,
                cloud_host=cloud_host if cloud_host else "api.trychroma.com",
            )
        else:
            persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_data")
            logger.warning(
                "No CHROMA_CLOUD_API_KEY found. Falling back to local PersistentClient: %s",
                persist_dir,
            )
            self._client = chromadb.PersistentClient(path=persist_dir)

        return self._client

    def get_collection(self) -> Collection:
        """Retrieve existing V2 collection without creating or mutating."""
        client = self.get_client()
        return client.get_collection(name=self.collection_name)

    def get_or_create_collection(self) -> Collection:
        """
        Get or safely create the V2 collection with cosine distance metric.
        Guarded against creating/overwriting V1 collection.
        """
        assert_safe_collection(self.collection_name, operation="get_or_create")
        client = self.get_client()
        return client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric},
        )

    def get_collection_count(self) -> int:
        """Return number of documents currently stored in the V2 collection."""
        try:
            col = self.get_collection()
            return col.count()
        except Exception:
            return 0

    def reset_v2_collection_safely(self, confirm: bool = False) -> None:
        """
        Safely reset the V2 collection.
        CANNOT be used on protected legacy collection under any circumstance.
        """
        assert_safe_collection(self.collection_name, operation="reset_v2_collection")

        if not confirm:
            raise ValueError(
                "Explicit confirmation required: pass confirm=True to reset V2 collection."
            )

        client = self.get_client()
        try:
            client.delete_collection(name=self.collection_name)
            logger.info("Deleted V2 collection: %s", self.collection_name)
        except Exception as exc:
            logger.info("Collection %s did not exist or already clean: %s", self.collection_name, exc)

        # Re-create empty
        client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric},
        )
        logger.info("Created clean V2 collection: %s", self.collection_name)

    def ingest_records(
        self,
        records: List[Dict[str, Any]],
        embeddings: List[List[float]],
        batch_size: int = 100,
    ) -> Tuple[int, int]:
        """
        Idempotently ingest records into InfyOne collection.

        Returns:
            Tuple[int, int]: (inserted_count, updated_or_skipped_count)
        """
        assert_safe_collection(self.collection_name, operation="ingest_records")

        if len(records) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(records)} records but {len(embeddings)} embeddings provided."
            )

        collection = self.get_or_create_collection()

        # Check existing IDs to ensure idempotency
        existing_count = collection.count()
        existing_ids = set()
        if existing_count > 0:
            existing_data = collection.get(include=[])
            existing_ids = set(existing_data.get("ids", []))
            logger.info(
                "Collection '%s' currently has %d existing records.",
                self.collection_name,
                existing_count,
            )

        # Prepare payloads
        ids_to_add: List[str] = []
        documents_to_add: List[str] = []
        metadatas_to_add: List[Dict[str, Any]] = []
        embeddings_to_add: List[List[float]] = []

        ids_to_update: List[str] = []
        documents_to_update: List[str] = []
        metadatas_to_update: List[Dict[str, Any]] = []
        embeddings_to_update: List[List[float]] = []

        for record, emb in zip(records, embeddings):
            doc_id = str(record["id"])
            doc_text = str(record["document"])

            # Transform metadata into Chroma-compatible dictionary
            metadata = {
                "document_type": str(record.get("document_type", "article")),
                "category": str(record.get("category", "general")),
                "chunk_index": int(record.get("chunk_index", 0)),
                "parent_doc_id": str(record.get("parent_doc_id", doc_id)),
                "total_chunks": int(record.get("total_chunks", 1)),
                "priority": int(record.get("priority", 0)),
                "tags": str(record.get("tags", "")),
                "created_at": str(record.get("created_at", "")),
                "title": str(record.get("title", "")),
            }

            if doc_id in existing_ids:
                ids_to_update.append(doc_id)
                documents_to_update.append(doc_text)
                metadatas_to_update.append(metadata)
                embeddings_to_update.append(emb)
            else:
                ids_to_add.append(doc_id)
                documents_to_add.append(doc_text)
                metadatas_to_add.append(metadata)
                embeddings_to_add.append(emb)

        # Batch insert new records
        inserted_count = 0
        for i in range(0, len(ids_to_add), batch_size):
            batch_slice = slice(i, i + batch_size)
            collection.add(
                ids=ids_to_add[batch_slice],
                documents=documents_to_add[batch_slice],
                metadatas=metadatas_to_add[batch_slice],
                embeddings=embeddings_to_add[batch_slice],
            )
            inserted_count += len(ids_to_add[batch_slice])

        # Batch update existing records (upsert behavior)
        updated_count = 0
        for i in range(0, len(ids_to_update), batch_size):
            batch_slice = slice(i, i + batch_size)
            collection.update(
                ids=ids_to_update[batch_slice],
                documents=documents_to_update[batch_slice],
                metadatas=metadatas_to_update[batch_slice],
                embeddings=embeddings_to_update[batch_slice],
            )
            updated_count += len(ids_to_update[batch_slice])

        logger.info(
            "Ingestion completed for '%s': %d inserted, %d updated.",
            self.collection_name,
            inserted_count,
            updated_count,
        )
        return inserted_count, updated_count

    def get_category_counts(self) -> Dict[str, int]:
        """Return counts of documents grouped by category in the V2 collection."""
        col = self.get_collection()
        all_meta = col.get(include=["metadatas"]).get("metadatas", [])
        counts: Dict[str, int] = {}
        for m in all_meta:
            cat = m.get("category", "unknown") if m else "unknown"
            counts[cat] = counts.get(cat, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))


@lru_cache()
def get_chroma_service_v2() -> ChromaServiceV2:
    """Return a cached singleton instance of ChromaServiceV2."""
    return ChromaServiceV2()
