# Interactive CLI chat loop (Feature 8): multi-turn session with query condensation,
# visible sources per answer, and a reset command.

from __future__ import annotations
from langchain_chroma import Chroma
from src.config import settings
from src.retrieval.vector_store import get_embedding_model
from src.retrieval.retriever import get_similarity_retriever
from src.chains.rag_chain import answer_question
from src.chains.memory import ChatSession, condense_query
from src.utils.logging_setup import configure_logging


def main() -> None:
    configure_logging()

    store = Chroma(
        collection_name=settings.vector_store_collection,
        embedding_function=get_embedding_model(),
        persist_directory=str(settings.vector_store_path),
    )
    retriever = get_similarity_retriever(store, k=settings.retrieval_k)
    session = ChatSession()

    print("=" * 60)
    print("SupportPearlz -- type your question, or 'reset' to clear history, 'exit' to quit.")
    print("=" * 60)

    while True:
        question = input("\nYou > ").strip()

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break
        if question.lower() == "reset":
            session.reset()
            print("[Session reset -- history cleared.]")
            continue

        standalone_question = condense_query(session, question)
        if standalone_question != question:
            print(f"[Rewritten query: {standalone_question!r}]")

        result, citations = answer_question(standalone_question, retriever)

        print(f"\nBot > {result.answer}")
        if citations:
            print("Sources:")
            for i, c in enumerate(citations, start=1):
                print(f"  [{i}] {c.source} -- {c.location}")
        print(f"Confidence: {result.confidence.value}")

        session.add_turn(question, result.answer)


if __name__ == "__main__":
    main()