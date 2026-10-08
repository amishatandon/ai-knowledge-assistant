from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L6-v2"
    ):

        print(
            f"Loading reranker model: "
            f"{model_name}"
        )

        self.model = CrossEncoder(
            model_name
        )

    def rerank(
        self,
        query,
        results,
        top_k=5
    ):

        if not results:

            return []

        pairs = []

        for result in results:

            pairs.append(
                (
                    query,
                    result["document"]
                )
            )

        scores = self.model.predict(
            pairs
        )

        reranked_results = []

        for i, result in enumerate(results):

            reranked_results.append({

                "id": result["id"],

                "document": result[
                    "document"
                ],

                "metadata": result[
                    "metadata"
                ],

                "rrf_score": result.get(
                    "rrf_score",
                    0
                ),

                "reranker_score": float(
                    scores[i]
                )
            })

        reranked_results.sort(
            key=lambda result:
                result["reranker_score"],
            reverse=True
        )

        return reranked_results[:top_k]
