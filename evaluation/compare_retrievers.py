# Task 06: compares retrieval strategies on a fixed set of test questions.
# Run with: python -m evaluation.compare_retrievers
from __future__ import annotations
from langchain_chroma import Chroma
from src.config import settings
from src.retrieval.vector_store import get_embedding_model
from src.retrieval.retriever import get_similarity_retriever, get_mmr_retriever, get_threshold_retriever
from src.utils.logging_setup import configure_logging
# Test questions with ground truth: which source document should answer them.
# You wrote these documents, so you know the correct answer's location.
TEST_QUESTIONS = [
    ("How long is the warranty on the AquaPearl 500 Pro?", "warranty_policy.md"),
    ("What's the return window for a refund?", "refund_policy.md"),
    ("How much does a replacement sediment filter cost?", "pricing_guide.csv"),
    # Hard: customer wording shares almost no keywords with the document text.
    ("my machine keeps beeping three times and then stops", "troubleshooting_guide.md"),
    # Hard: real answer is in the Installation Guide, not the more obvious Troubleshooting Guide.
    ("water tastes weird after I changed the filter", "installation_guide.pdf"),
    # Hard: this is the manual's own multi-document seam test case.
    ("my unit leaked and damaged my kitchen cabinet, is that covered and how do I claim?", "warranty_policy.md"),
    ("how do I claim a warranty repair", "service_maintenance_agreement.pdf"),
    ("what data do you collect through the app", "privacy_policy.md"),
    # Hard: near-identical alert codes (E-03, E-05) are easy for embeddings to confuse.
    ("what does error E-05 mean on firmware 2.3", "product_changelog.md"),
    # Genuinely unanswerable -- included so we can see how each strategy handles it.
    ("do you offer student discounts", None),
]

def hit_at_k(retriever, question: str, expected_source: str | None) -> bool:
    # For unanswerable questions (expected_source=None), we don't score hit/miss here --
    # that's covered properly by the refusal-rate test in Task 11's evaluation harness.
    if expected_source is None:
        return None
    results = retriever.invoke(question)
    sources = [doc.metadata.get("source", "") for doc in results]
    return expected_source in sources

def show_misses(retriever, name: str) -> None:
    # Print exactly which questions this strategy got wrong, and what it retrieved instead.
    print(f"\n--- Misses for {name} ---")
    for question, expected_source in TEST_QUESTIONS:
        if expected_source is None:
            continue
        results = retriever.invoke(question)
        sources = [doc.metadata.get("source", "") for doc in results]
        if expected_source not in sources:
            print(f"  Q: {question!r}")
            print(f"  Expected: {expected_source} | Got: {sources}")

def calibrate_threshold(store) -> None:
    # Task 06: find where relevant-question scores and irrelevant-question scores separate.
    print("\n--- Threshold Calibration ---")
    relevant_questions = [q for q, src in TEST_QUESTIONS if src is not None]
    irrelevant_questions = ["do you offer student discounts", "is the filter NSF certified", "what's the weather today"]

    print("Scores for ANSWERABLE questions (top-1 hit, lower = more similar):")
    relevant_scores = []
    for q in relevant_questions:
        results = store.similarity_search_with_score(q, k=1)
        if results:
            score = results[0][1]
            relevant_scores.append(score)
            print(f"  {score:.4f}  | {q!r}")

    print("\nScores for UNANSWERABLE questions (top-1 hit, lower = more similar):")
    irrelevant_scores = []
    for q in irrelevant_questions:
        results = store.similarity_search_with_score(q, k=1)
        if results:
            score = results[0][1]
            irrelevant_scores.append(score)
            print(f"  {score:.4f}  | {q!r}")

    if relevant_scores and irrelevant_scores:
        max_relevant = max(relevant_scores)
        min_irrelevant = min(irrelevant_scores)
        print(f"\nMax score among answerable questions : {max_relevant:.4f}")
        print(f"Min score among unanswerable questions: {min_irrelevant:.4f}")
        if max_relevant < min_irrelevant:
            suggested = (max_relevant + min_irrelevant) / 2
            print(f"Clean separation found. Suggested threshold: {suggested:.4f}")
        else:
            print("WARNING: scores overlap -- no clean threshold separates answerable from unanswerable.")

def main() -> None:
    configure_logging()
    store = Chroma(
        collection_name=settings.vector_store_collection,
        embedding_function=get_embedding_model(),
        persist_directory=str(settings.vector_store_path),
    )
    calibrate_threshold(store)
    strategies = {
        "similarity_k3": get_similarity_retriever(store, k=3),
        "similarity_k8": get_similarity_retriever(store, k=8),
        "mmr_k4": get_mmr_retriever(store, k=4),
    }

    print(f"{'Strategy':<18} | Hit@k | Score")
    print("-" * 40)
    for name, retriever in strategies.items():
        hits = 0
        scored_questions = 0
        for question, expected_source in TEST_QUESTIONS:
            result = hit_at_k(retriever, question, expected_source)
            if result is None:
                continue  # unanswerable question, not scored here
            scored_questions += 1
            if result:
                hits += 1
        score = hits / scored_questions if scored_questions else 0
        print(f"{name:<18} | {hits}/{scored_questions}  | {score:.2f}")
        show_misses(retriever, name)

if __name__ == "__main__":
    main()
