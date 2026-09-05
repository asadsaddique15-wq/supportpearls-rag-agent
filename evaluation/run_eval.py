# Task 11: runs the full test_questions.json set and writes results to evaluation/results/.
# Run with: python -m evaluation.run_eval

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from langchain_chroma import Chroma
from src.config import settings
from src.retrieval.vector_store import get_embedding_model
from src.retrieval.retriever import get_similarity_retriever
from src.chains.rag_chain import answer_question
from src.utils.logging_setup import configure_logging

TEST_FILE = Path("evaluation/test_questions.json")
RESULTS_DIR = Path("evaluation/results")


def load_test_cases() -> list[dict]:
    with open(TEST_FILE, encoding="utf-8") as f:
        return json.load(f)


def run_case(case: dict, retriever) -> dict:
    # Run one test case through the chain and record everything needed for manual scoring.
    result, citations = answer_question(case["question"], retriever)

    cited_sources = [c.source for c in citations]
    ground_truth = case.get("ground_truth_source")
    expected_sources = [s.strip() for s in ground_truth.split(",")] if ground_truth else []

    # Automatic signal only -- correctness/groundedness still need a human eye (per manual's instructions).
    source_match = any(s in cited_sources for s in expected_sources) if expected_sources else (not result.answered)

    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "expected_behavior": case["expected_behavior"],
        "expected_sources": expected_sources,
        "answer": result.answer,
        "cited_sources": cited_sources,
        "confidence": result.confidence.value,
        "answered": result.answered,
        "source_match_auto_check": source_match,
        # Left blank for manual scoring, per the manual's expectation that correctness/
        # groundedness/citation accuracy be manually verified against a written rubric.
        "manual_verdict": None,
        "manual_notes": "",
    }


def main() -> None:
    configure_logging()
    cases = load_test_cases()

    store = Chroma(
        collection_name=settings.vector_store_collection,
        embedding_function=get_embedding_model(),
        persist_directory=str(settings.vector_store_path),
    )
    retriever = get_similarity_retriever(store, k=settings.retrieval_k)

    results = []
    for case in cases:
        print(f"Running {case['id']}: {case['question'][:60]}...")
        results.append(run_case(case, retriever))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f"eval_results_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Readable table
    table_path = RESULTS_DIR / f"eval_results_{timestamp}.txt"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(f"{'ID':<5} {'Category':<12} {'Answered':<9} {'Confidence':<10} {'Auto-Match':<10} Question\n")
        f.write("-" * 100 + "\n")
        for r in results:
            f.write(
                f"{r['id']:<5} {r['category']:<12} {str(r['answered']):<9} "
                f"{r['confidence']:<10} {str(r['source_match_auto_check']):<10} {r['question'][:50]}\n"
            )

    # Aggregate metrics
    unanswerable_cases = [r for r in results if r["category"] == "unanswerable"]
    refusal_rate = (
        sum(1 for r in unanswerable_cases if not r["answered"]) / len(unanswerable_cases)
        if unanswerable_cases else 0
    )
    auto_match_rate = sum(1 for r in results if r["source_match_auto_check"]) / len(results)

    print(f"\n{'='*60}")
    print(f"Results written to: {json_path}")
    print(f"Readable table: {table_path}")
    print(f"Refusal rate on unanswerable cases: {refusal_rate:.2f} ({sum(1 for r in unanswerable_cases if not r['answered'])}/{len(unanswerable_cases)})")
    print(f"Auto source-match rate (all cases): {auto_match_rate:.2f}")
    print("NOTE: correctness, groundedness, and citation accuracy still require your manual review")
    print("      per the written rubric -- open the JSON file and fill in manual_verdict/manual_notes.")


if __name__ == "__main__":
    main()