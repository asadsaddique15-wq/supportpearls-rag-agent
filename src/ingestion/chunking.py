# Chunking module (Task 03): splits documents into retrieval-sized chunks.
from __future__ import annotations
import logging
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
logger = logging.getLogger(__name__)
# Doc types that are already atomic (one CSV row = one fact) and should NOT be split further.
NO_SPLIT_DOC_TYPES = {"pricing"}

def chunk_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    # Split a list of loaded documents into chunks, preserving metadata on every chunk.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],  # try headings first, then paragraphs, then sentences
    )
    chunks: list[Document] = []
    for doc in documents:
        doc_type = doc.metadata.get("doc_type", "unknown")
        if doc_type in NO_SPLIT_DOC_TYPES:
            # CSV rows / already-atomic content: pass through unchanged.
            chunks.append(doc)
            continue
        split_docs = splitter.split_documents([doc])  # split_documents copies metadata to every resulting chunk
        chunks.extend(split_docs)

    logger.info(
        "Chunking complete: %d input documents -> %d chunks (chunk_size=%d, overlap=%d)",
        len(documents), len(chunks), chunk_size, chunk_overlap,
    )
    return chunks

def chunk_stats(chunks: list[Document]) -> dict:
    # Quick stats used in the Task 03 comparison table: count, mean length, max length.
    lengths = [len(c.page_content) for c in chunks]
    if not lengths:
        return {"count": 0, "mean_length": 0, "max_length": 0}
    return {
        "count": len(lengths),
        "mean_length": round(sum(lengths) / len(lengths), 1),
        "max_length": max(lengths),
    }