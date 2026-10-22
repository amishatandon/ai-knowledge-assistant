from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L6-v2"
    ):

        print(
            f"Loading reranker model: {model_name}"
        )

        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query,
        results,
        top_k=5
    ):

        if not results:
            return []

        # Create query-document pairs for the Cross-Encoder.
        pairs = [
            (query, result["document"])
            for result in results
        ]

        # Calculate relevance scores.
        scores = self.model.predict(pairs)

        reranked_results = []

        for i, result in enumerate(results):

            reranked_results.append({
                "id": result["id"],

                "document": result["document"],

                "metadata": result["metadata"],

                # Preserve the RRF score.
                "rrf_score": result.get(
                    "rrf_score",
                    0.0
                ),

                # Preserve the MMR score.
                "mmr_score": result.get(
                    "mmr_score"
                ),

                # Add the Cross-Encoder relevance score.
                "reranker_score": float(scores[i])
            })

        # Sort by Cross-Encoder relevance, highest first.
        reranked_results.sort(
            key=lambda result: result["reranker_score"],
            reverse=True
        )

        return reranked_results[:top_k]
