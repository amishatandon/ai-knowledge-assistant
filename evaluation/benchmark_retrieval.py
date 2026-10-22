import json
from pathlib import Path

from rag.embeddings import EmbeddingModel
from rag.vectorstore import VectorStore
from rag.bm25 import BM25Retriever
from rag.reranker import Reranker


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

    return (
        metadata.get("source", "")
        in expected_sources
    )


def get_first_relevant_rank(
    results,
    expected_sources
):

    for rank, result in enumerate(
        results,
        start=1
    ):

        if source_matches(
            result["metadata"],
            expected_sources
        ):

            return rank

    return None


def calculate_metrics(
    ranks,
    total_questions
):

    recall_at_1 = sum(
        1
        for rank in ranks
        if rank is not None
        and rank <= 1
    ) / total_questions

    recall_at_3 = sum(
        1
        for rank in ranks
        if rank is not None
        and rank <= 3
    ) / total_questions

    recall_at_5 = sum(
        1
        for rank in ranks
        if rank is not None
        and rank <= 5
    ) / total_questions

    reciprocal_rank_sum = sum(
        (
            1 / rank
            if rank is not None
            else 0
        )
        for rank in ranks
    )

    mrr = (
        reciprocal_rank_sum
        / total_questions
    )

    return {
        "Recall@1": recall_at_1,
        "Recall@3": recall_at_3,
        "Recall@5": recall_at_5,
        "MRR": mrr
    }


def dense_search(
    embedding_model,
    vector_store,
    query,
    top_k=5
):

    query_embedding = (
        embedding_model.embed_text(
            query
        )
    )

    results = vector_store.search(
        query_embedding,
        top_k=top_k
    )

    documents = results[
        "documents"
    ][0]

    metadatas = results[
        "metadatas"
    ][0]

    ids = results[
        "ids"
    ][0]

    dense_results = []

    for i in range(len(documents)):

        dense_results.append({

            "id": ids[i],

            "document": documents[i],

            "metadata": metadatas[i],

            "rank": i + 1

        })

    return dense_results


def bm25_search(
    bm25_retriever,
    query,
    top_k=5
):

    return bm25_retriever.search(
        query,
        top_k=top_k
    )


def rrf_fusion(
    dense_results,
    bm25_results,
    k=60
):

    fused_scores = {}

    result_data = {}

    for result in dense_results:

        document_id = result["id"]

        rank = result["rank"]

        score = 1 / (k + rank)

        fused_scores[
            document_id
        ] = fused_scores.get(
            document_id,
            0
        ) + score

        result_data[
            document_id
        ] = result

    for result in bm25_results:

        document_id = result["id"]

        rank = result["rank"]

        score = 1 / (k + rank)

        fused_scores[
            document_id
        ] = fused_scores.get(
            document_id,
            0
        ) + score

        if document_id not in result_data:

            result_data[
                document_id
            ] = result

    ranked_ids = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True
    )

    fused_results = []

    for rank, document_id in enumerate(
        ranked_ids,
        start=1
    ):

        result = result_data[
            document_id
        ]

        fused_results.append({

            "id": document_id,

            "document": result[
                "document"
            ],

            "metadata": result[
                "metadata"
            ],

            "rrf_score": fused_scores[
                document_id
            ],

            "rank": rank

        })

    return fused_results


def rerank(
    reranker,
    query,
    results,
    top_k=5
):

    # Rerank a larger candidate pool
    candidates = results[:20]

    pairs = [
        (
            query,
            result["document"]
        )
        for result in candidates
    ]

    scores = reranker.model.predict(
        pairs
    )

    reranked = []

    for i, result in enumerate(
        candidates
    ):

        reranked.append({

            "id": result["id"],

            "document": result[
                "document"
            ],

            "metadata": result[
                "metadata"
            ],

            "reranker_score": float(
                scores[i]
            )

        })

    reranked.sort(
        key=lambda result:
            result["reranker_score"],
        reverse=True
    )

    return reranked[:top_k]


def evaluate_method(
    method_name,
    questions,
    retrieval_function
):

    print("\n")
    print("=" * 70)
    print(method_name)
    print("=" * 70)

    ranks = []

    for index, item in enumerate(
        questions,
        start=1
    ):

        question = item[
            "question"
        ]

        expected_sources = item[
            "expected_sources"
        ]

        results = retrieval_function(
            question
        )

        rank = get_first_relevant_rank(
            results,
            expected_sources
        )

        ranks.append(rank)

        print(
            f"{index}. "
            f"Rank = {rank}"
        )

    return calculate_metrics(
        ranks,
        len(questions)
    )


def main():

    questions = load_questions()

    print(
        f"Loaded {len(questions)} "
        f"evaluation questions."
    )

    # ------------------------------------------------
    # Load models ONCE
    # ------------------------------------------------

    embedding_model = EmbeddingModel()

    vector_store = VectorStore()

    bm25 = BM25Retriever(
        vector_store
    )

    reranker = Reranker()

    # ------------------------------------------------
    # 1. Dense
    # ------------------------------------------------

    dense_metrics = evaluate_method(

        "1. DENSE RETRIEVAL",

        questions,

        lambda query:
            dense_search(
                embedding_model,
                vector_store,
                query,
                top_k=5
            )

    )

    # ------------------------------------------------
    # 2. BM25
    # ------------------------------------------------

    bm25_metrics = evaluate_method(

        "2. BM25",

        questions,

        lambda query:
            bm25_search(
                bm25,
                query,
                top_k=5
            )

    )

    # ------------------------------------------------
    # 3. Hybrid + RRF
    # ------------------------------------------------

    def hybrid_search(query):

        dense_results = dense_search(

            embedding_model,

            vector_store,

            query,

            top_k=20

        )

        bm25_results = bm25_search(

            bm25,

            query,

            top_k=20

        )

        fused_results = rrf_fusion(

            dense_results,

            bm25_results

        )

        return fused_results[:5]

    hybrid_metrics = evaluate_method(

        "3. HYBRID + RRF",

        questions,

        hybrid_search

    )

    # ------------------------------------------------
    # 4. Hybrid + RRF + Reranker
    # ------------------------------------------------

    def reranked_search(query):

        dense_results = dense_search(

            embedding_model,

            vector_store,

            query,

            top_k=20

        )

        bm25_results = bm25_search(

            bm25,

            query,

            top_k=20

        )

        fused_results = rrf_fusion(

            dense_results,

            bm25_results

        )

        return rerank(

            reranker,

            query,

            fused_results,

            top_k=5

        )

    reranked_metrics = evaluate_method(

        "4. HYBRID + RRF + RERANKER",

        questions,

        reranked_search

    )

    # ------------------------------------------------
    # Final comparison
    # ------------------------------------------------

    print("\n")

    print("=" * 70)

    print(
        "FINAL RETRIEVAL COMPARISON"
    )

    print("=" * 70)

    print(
        f"{'Method':<30}"
        f"{'R@1':<10}"
        f"{'R@3':<10}"
        f"{'R@5':<10}"
        f"{'MRR':<10}"
    )

    print("-" * 70)

    results = [

        (
            "Dense",
            dense_metrics
        ),

        (
            "BM25",
            bm25_metrics
        ),

        (
            "Hybrid + RRF",
            hybrid_metrics
        ),

        (
            "Hybrid + RRF + Reranker",
            reranked_metrics
        )

    ]

    for name, metrics in results:

        print(
            f"{name:<30}"
            f"{metrics['Recall@1']:<10.3f}"
            f"{metrics['Recall@3']:<10.3f}"
            f"{metrics['Recall@5']:<10.3f}"
            f"{metrics['MRR']:<10.3f}"
        )


if __name__ == "__main__":

    main()