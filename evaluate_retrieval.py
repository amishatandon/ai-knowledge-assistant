import json
import time
from pathlib import Path
from statistics import mean

from rag.retriever import Retriever


DATASET_PATH = Path(__file__).parent / "evaluation_dataset.json"
OUTPUT_PATH = Path(__file__).parent / "evaluation_results.json"
TOP_K = 5


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not dataset:
        raise ValueError("Evaluation dataset is empty.")

    for index, item in enumerate(dataset, start=1):
        if not item.get("question"):
            raise ValueError(f"Missing question in entry {index}.")

        if not item.get("relevant_doc_ids"):
            raise ValueError(
                f"Missing relevant_doc_ids in entry {index}."
            )

    return dataset


def install_rewrite_timer(retriever):
    """
    Time the existing rewrite method without modifying
    rag/retriever.py.

    The wrapper measures only the time spent inside
    _rewrite_query(), including Gemini API delays/retries.
    """

    original_method = retriever._rewrite_query
    retriever.last_rewrite_seconds = 0.0

    def timed_rewrite(query):
        start = time.perf_counter()

        try:
            return original_method(query)
        finally:
            retriever.last_rewrite_seconds = (
                time.perf_counter() - start
            )

    retriever._rewrite_query = timed_rewrite


def calculate_metrics(retrieved_ids, relevant_ids):
    retrieved_ids = retrieved_ids[:TOP_K]
    relevant_ids = set(relevant_ids)

    relevant_retrieved = (
        set(retrieved_ids) & relevant_ids
    )

    recall = (
        len(relevant_retrieved) / len(relevant_ids)
        if relevant_ids
        else 0.0
    )

    reciprocal_rank = 0.0

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            reciprocal_rank = 1.0 / rank
            break

    return recall, reciprocal_rank


def evaluate_mode(retriever, dataset, query_rewriting):
    retriever.query_rewriting = query_rewriting

    mode_name = (
        "WITH QUERY REWRITING"
        if query_rewriting
        else "WITHOUT QUERY REWRITING"
    )

    print("\n" + "=" * 75)
    print(mode_name)
    print("=" * 75)

    question_results = []

    for index, item in enumerate(dataset, start=1):
        question = item["question"]
        relevant_ids = item["relevant_doc_ids"]

        print(
            f"\nQuestion {index}/{len(dataset)}: {question}"
        )

        retriever.last_rewrite_seconds = 0.0

        total_start = time.perf_counter()

        results = retriever.retrieve(
            question,
            top_k=TOP_K
        )

        total_seconds = (
            time.perf_counter() - total_start
        )

        rewrite_seconds = retriever.last_rewrite_seconds

        # This includes dense retrieval, BM25, RRF, MMR,
        # reranking, and pipeline overhead after rewriting.
        retrieval_seconds = max(
            0.0,
            total_seconds - rewrite_seconds
        )

        retrieved_ids = results["ids"][0]

        recall, reciprocal_rank = calculate_metrics(
            retrieved_ids,
            relevant_ids
        )

        question_result = {
            "question": question,
            "retrieved_ids": retrieved_ids,
            "relevant_doc_ids": relevant_ids,
            "recall_at_5": recall,
            "reciprocal_rank": reciprocal_rank,
            "rewrite_seconds": rewrite_seconds,
            "retrieval_seconds": retrieval_seconds,
            "total_seconds": total_seconds
        }

        question_results.append(question_result)

        print("\nTop retrieved IDs:")

        relevant_set = set(relevant_ids)

        for rank, doc_id in enumerate(
            retrieved_ids, start=1
        ):
            marker = (
                "RELEVANT"
                if doc_id in relevant_set
                else ""
            )
            print(f"  {rank}. {doc_id} {marker}")

        print(f"\nRecall@{TOP_K}: {recall:.3f}")
        print(f"MRR contribution: {reciprocal_rank:.3f}")
        print(f"Rewrite time: {rewrite_seconds:.2f} seconds")
        print(f"Retrieval pipeline time: {retrieval_seconds:.2f} seconds")
        print(f"Total time: {total_seconds:.2f} seconds")

    summary = {
        "mode": mode_name,
        "number_of_questions": len(question_results),
        "mean_recall_at_5": mean(
            r["recall_at_5"] for r in question_results
        ),
        "mrr": mean(
            r["reciprocal_rank"] for r in question_results
        ),
        "mean_rewrite_seconds": mean(
            r["rewrite_seconds"] for r in question_results
        ),
        "mean_retrieval_seconds": mean(
            r["retrieval_seconds"] for r in question_results
        ),
        "mean_total_seconds": mean(
            r["total_seconds"] for r in question_results
        ),
        "question_results": question_results
    }

    return summary


def print_comparison(baseline, rewritten):
    print("\n\n" + "=" * 90)
    print("FINAL EVALUATION COMPARISON")
    print("=" * 90)

    print(
        f"\n{'Metric':<30}"
        f"{'Without rewriting':>22}"
        f"{'With rewriting':>22}"
    )
    print("-" * 74)

    metrics = [
        (
            "Mean Recall@5",
            "mean_recall_at_5",
            ".3f"
        ),
        (
            "MRR",
            "mrr",
            ".3f"
        ),
        (
            "Mean rewrite time (sec)",
            "mean_rewrite_seconds",
            ".2f"
        ),
        (
            "Mean retrieval time (sec)",
            "mean_retrieval_seconds",
            ".2f"
        ),
        (
            "Mean total time (sec)",
            "mean_total_seconds",
            ".2f"
        )
    ]

    for label, key, number_format in metrics:
        left = format(baseline[key], number_format)
        right = format(rewritten[key], number_format)

        print(f"{label:<30}{left:>22}{right:>22}")

    print("-" * 74)

    recall_change = (
        rewritten["mean_recall_at_5"]
        - baseline["mean_recall_at_5"]
    )

    print(f"\nRecall@5 change: {recall_change:+.3f}")

    print(
        "Total latency change: "
        f"{rewritten['mean_total_seconds'] - baseline['mean_total_seconds']:+.2f} seconds"
    )

    print(
        "\nReminder: baseline-derived relevance labels may favor "
        "the no-rewriting configuration. Verify the labels before "
        "drawing conclusions."
    )


def main():
    print("=" * 75)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 75)

    dataset = load_dataset()

    print(f"\nLoaded {len(dataset)} evaluation questions.")
    print(f"Evaluating the top {TOP_K} retrieved chunks.")

    # Initialize models only once.
    retriever = Retriever(query_rewriting=True)

    install_rewrite_timer(retriever)

    baseline = evaluate_mode(
        retriever,
        dataset,
        query_rewriting=False
    )

    rewritten = evaluate_mode(
        retriever,
        dataset,
        query_rewriting=True
    )

    print_comparison(baseline, rewritten)

    output = {
        "top_k": TOP_K,
        "baseline": baseline,
        "query_rewriting": rewritten
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(f"\nDetailed results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()