import json
from pathlib import Path

from rag.embeddings import EmbeddingModel
from rag.vectorstore import VectorStore
from rag.bm25 import BM25Retriever
from rag.retriever import Retriever


EVALUATION_FILE = (
    Path(__file__).parent
    / "retrieval_questions.json"
)


def load_questions():

    with open(
        EVALUATION_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def source_matches(
    metadata,
    expected_sources
):

    source = metadata.get(
        "source",
        ""
    )

    return source in expected_sources


def evaluate():

    questions = load_questions()

    print(
        f"\nLoaded {len(questions)} "
        f"evaluation questions.\n"
    )

    print("=" * 70)

    # ------------------------------------------------
    # Load retrieval components
    # ------------------------------------------------

    embedding_model = EmbeddingModel()

    vector_store = VectorStore()

    bm25_retriever = BM25Retriever(
        vector_store
    )

    # Our complete retrieval pipeline
    retriever = Retriever()

    # ------------------------------------------------
    # Metrics
    # ------------------------------------------------

    recall_at_1 = 0
    recall_at_3 = 0
    recall_at_5 = 0

    reciprocal_rank_sum = 0

    # ------------------------------------------------
    # Evaluate each question
    # ------------------------------------------------

    for question_number, item in enumerate(
        questions,
        start=1
    ):

        question = item["question"]

        expected_sources = item[
            "expected_sources"
        ]

        print(
            f"\nQuestion {question_number}:"
        )

        print(question)

        print(
            f"Expected: "
            f"{expected_sources}"
        )

        # --------------------------------------------
        # Run our current retrieval pipeline
        # --------------------------------------------

        results = retriever.retrieve(
            question,
            top_k=5
        )

        documents = results[
            "documents"
        ][0]

        metadatas = results[
            "metadatas"
        ][0]

        # --------------------------------------------
        # Display retrieved documents
        # --------------------------------------------

        first_relevant_rank = None

        for rank, metadata in enumerate(
            metadatas,
            start=1
        ):

            source = metadata.get(
                "source"
            )

            page = metadata.get(
                "page"
            )

            print(
                f"  {rank}. "
                f"{source} "
                f"(page {page})"
            )

            if (
                first_relevant_rank is None
                and source_matches(
                    metadata,
                    expected_sources
                )
            ):

                first_relevant_rank = rank

        # --------------------------------------------
        # Recall@K
        # --------------------------------------------

        if first_relevant_rank is not None:

            if first_relevant_rank <= 1:
                recall_at_1 += 1

            if first_relevant_rank <= 3:
                recall_at_3 += 1

            if first_relevant_rank <= 5:
                recall_at_5 += 1

            # ----------------------------------------
            # Reciprocal Rank
            # ----------------------------------------

            reciprocal_rank_sum += (
                1 / first_relevant_rank
            )

        print(
            f"First relevant rank: "
            f"{first_relevant_rank}"
        )

    # ------------------------------------------------
    # Calculate final metrics
    # ------------------------------------------------

    total_questions = len(
        questions
    )

    recall_1 = (
        recall_at_1
        / total_questions
    )

    recall_3 = (
        recall_at_3
        / total_questions
    )

    recall_5 = (
        recall_at_5
        / total_questions
    )

    mrr = (
        reciprocal_rank_sum
        / total_questions
    )

    # ------------------------------------------------
    # Print results
    # ------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 70)

    print(
        f"Recall@1 : {recall_1:.3f}"
    )

    print(
        f"Recall@3 : {recall_3:.3f}"
    )

    print(
        f"Recall@5 : {recall_5:.3f}"
    )

    print(
        f"MRR      : {mrr:.3f}"
    )

    print("=" * 70)


if __name__ == "__main__":

    evaluate()