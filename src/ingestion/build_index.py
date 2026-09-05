# CLI entry point: ingestion -> chunking -> embedding -> vector store persistence.
# Run with: python -m src.ingestion.build_index
# Force a full rebuild with: python -m src.ingestion.build_index --rebuild

from __future__ import annotations
import argparse
import logging
from src.config import settings
from src.ingestion.loaders import load_knowledge_base
from src.ingestion.chunking import chunk_documents, chunk_stats
from src.retrieval.vector_store import build_vector_store, similarity_search_debug
from src.utils.logging_setup import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Force a full re-embed even if a store already exists.")
    args = parser.parse_args()

    configure_logging()
    logger.info("Starting ingestion from %s", settings.knowledge_base_dir)

    result = load_knowledge_base(settings.knowledge_base_dir)
    print(result.summary())

    if not result.documents:
        logger.error("No documents were successfully loaded. Aborting.")
        raise SystemExit(1)

    chunks = chunk_documents(result.documents, settings.chunk_size, settings.chunk_overlap)
    stats = chunk_stats(chunks)
    print(f"\nChunking stats: {stats}")

    store = build_vector_store(chunks, rebuild=args.rebuild)

    # Sanity-check query proving the index actually works end to end.
    similarity_search_debug(store, "how long is the warranty", k=3)


if __name__ == "__main__":
    main()