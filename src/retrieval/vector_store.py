# Vector store module (Task 05): create, persist, and reload the Chroma index.
# Critical requirement: query-time startup must NOT re-embed if a valid store already exists.

from __future__ import annotations
import logging
from pathlib import Path
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from src.config import settings

logger = logging.getLogger(__name__)


def get_embedding_model() -> OpenAIEmbeddings:
    # Single source of truth for the embedding model — used identically for indexing and querying.
    return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)


def _store_exists(path: Path) -> bool:
    # A Chroma store that has actually been persisted will contain a sqlite file.
    return path.exists() and any(path.glob("*.sqlite3"))


def build_vector_store(chunks: list[Document], rebuild: bool = False) -> Chroma:
    # Load an existing persisted store, or build a new one from chunks if none exists (or --rebuild is forced).
    path = settings.vector_store_path
    embeddings = get_embedding_model()

    if _store_exists(path) and not rebuild:
        logger.info("Existing vector store found at %s — loading from disk (NOT re-embedding).", path)
        return Chroma(
            collection_name=settings.vector_store_collection,
            embedding_function=embeddings,
            persist_directory=str(path),
        )

    if rebuild and _store_exists(path):
        logger.info("--rebuild flag set — deleting existing store at %s before rebuilding.", path)
        import shutil
        shutil.rmtree(path)

    logger.info("Building new vector store from %d chunks (this calls the embedding API).", len(chunks))
    path.mkdir(parents=True, exist_ok=True)
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=settings.vector_store_collection,
        persist_directory=str(path),
    )
    logger.info("Vector store built and persisted to %s", path)
    return store


def similarity_search_debug(store: Chroma, query: str, k: int) -> None:
    # Debugging helper (Task 05 requirement): print score, source, and a content preview for each hit.
    results = store.similarity_search_with_score(query, k=k)
    print(f"\nQuery: {query!r}  (top {k})")
    print("-" * 60)
    for doc, score in results:
        source = doc.metadata.get("source", "unknown")
        location = doc.metadata.get("location", "")
        preview = doc.page_content[:200].replace("\n", " ")
        print(f"score={score:.4f} | source={source} ({location})\n  {preview}...\n")
        