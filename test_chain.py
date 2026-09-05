# Quick manual smoke test for the RAG chain -- not part of the formal test suite.
from langchain_chroma import Chroma
from src.config import settings
from src.retrieval.vector_store import get_embedding_model
from src.retrieval.retriever import get_similarity_retriever
from src.chains.rag_chain import answer_question
from src.utils.logging_setup import configure_logging
configure_logging()

store = Chroma(
    collection_name=settings.vector_store_collection,
    embedding_function=get_embedding_model(),
    persist_directory=str(settings.vector_store_path),
)
retriever = get_similarity_retriever(store, k=8)  # k=8 won Task 06's comparison
questions = [
    "How long is the warranty on the AquaPearl 500 Pro?",
    "Do you offer student discounts?",
]

for q in questions:
    print(f"\n{'='*60}\nQ: {q}")
    result = answer_question(q, retriever)
    print(f"A: {result.answer}")
    print(f"Sources: {result.sources}")
    print(f"Confidence: {result.confidence} | Answered: {result.answered}")