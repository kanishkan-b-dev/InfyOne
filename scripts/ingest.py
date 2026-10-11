#!/usr/bin/env python
"""
Document ingestion script for Athena Institute of Technology AI Chatbot.

Usage:
    python scripts/ingest.py [--data-dir ./data/documents] [--force]

Options:
    --data-dir   Path to directory containing documents (default: ./data/documents)
    --force      Re-ingest all documents even if already indexed
    --verbose    Enable verbose/debug logging
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path so we can import app modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.rag.ingestion import ingest_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest documents into ChromaDB for the Athena AI Chatbot."
    )
    parser.add_argument(
        "--data-dir",
        default="./data/documents",
        help="Path to the documents directory (default: ./data/documents)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingestion even if documents already exist in ChromaDB",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()

    if args.verbose:
        import os
        os.environ["DEBUG"] = "true"

    setup_logging()
    logger = logging.getLogger(__name__)

    data_dir = Path(args.data_dir).resolve()
    logger.info("=" * 60)
    logger.info("Athena Institute AI Chatbot — Document Ingestion")
    logger.info("=" * 60)
    logger.info("Data directory : %s", data_dir)
    logger.info("ChromaDB path  : %s", settings.CHROMA_PERSIST_DIRECTORY)
    logger.info("Collection     : %s", settings.CHROMA_COLLECTION_NAME)
    logger.info("Force re-ingest: %s", args.force)
    logger.info("-" * 60)

    if not data_dir.exists():
        logger.error("Data directory does not exist: %s", data_dir)
        logger.info("Create the directory and add your documents, then re-run.")
        sys.exit(1)

    try:
        summary = ingest_directory(str(data_dir), force=args.force)
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc, exc_info=True)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Ingestion Summary")
    logger.info("=" * 60)

    if not summary:
        logger.warning("No documents were processed.")
    else:
        for filename, count in summary.items():
            status = f"{count} new chunks" if count > 0 else "already indexed (skipped)"
            logger.info("  %-40s %s", filename, status)

    total_new = sum(summary.values())
    logger.info("-" * 60)
    logger.info("Total new chunks added: %d", total_new)

    # Show collection size after ingestion
    try:
        from app.rag.vectorstore import get_collection_count
        total = get_collection_count()
        logger.info("Total documents in ChromaDB collection: %d", total)
    except Exception:
        pass

    logger.info("=" * 60)
    logger.info("Ingestion complete!")


if __name__ == "__main__":
    main()
