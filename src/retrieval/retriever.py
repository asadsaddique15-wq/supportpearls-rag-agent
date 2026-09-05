# Retriever construction (Task 06): builds different retrieval strategies from the vector store.
from __future__ import annotations
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

def get_similarity_retriever(store: Chroma, k: int) -> VectorStoreRetriever:
    # Plain similarity search — returns the k closest chunks by distance.
    return store.as_retriever(search_type="similarity", search_kwargs={"k": k})

def get_mmr_retriever(store: Chroma, k: int, fetch_k: int = 20) -> VectorStoreRetriever:
    # MMR trades some relevance for diversity, avoiding near-duplicate chunks in the top-k.
    return store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": 0.5},
    )

def get_threshold_retriever(store: Chroma, k: int, score_threshold: float) -> VectorStoreRetriever:
    # Only returns chunks scoring above the threshold; can return fewer than k (or zero).
    return store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": score_threshold},
    )