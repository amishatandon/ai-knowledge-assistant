
from sklearn.metrics.pairwise import cosine_similarity


class MMR:

    def __init__(self, embedding_model, lambda_param=0.7):
        if not 0 <= lambda_param <= 1:
            raise ValueError(
                "lambda_param must be between 0 and 1"
            )

        self.embedding_model = embedding_model
        self.lambda_param = lambda_param

    def select(self, query, candidates, top_k=10):

        if not candidates:
            return []

        if top_k <= 0:
            return []

        # Encode the query and candidate documents.
        query_embedding = self.embedding_model.embed_text(query)

        candidate_documents = [
            {"text": candidate["document"]}
            for candidate in candidates
        ]

        candidate_embeddings = (
            self.embedding_model.embed_documents(
                candidate_documents
            )
        )

        # Relevance: similarity between each candidate and the query.
        query_similarities = cosine_similarity(
            candidate_embeddings,
            query_embedding.reshape(1, -1)
        ).flatten()

        # Redundancy: similarity between candidate documents.
        document_similarities = cosine_similarity(
            candidate_embeddings
        )

        selected_indices = []
        remaining_indices = list(range(len(candidates)))
        mmr_scores = {}

        # The loop naturally stops when either:
        # 1. top_k candidates have been selected, or
        # 2. no candidates remain.
        while remaining_indices and len(selected_indices) < top_k:

            best_index = None
            best_score = -float("inf")

            for index in remaining_indices:

                relevance = query_similarities[index]

                if not selected_indices:
                    redundancy = 0.0
                else:
                    redundancy = max(
                        document_similarities[
                            index, selected_indices
                        ]
                    )

                score = (
                    self.lambda_param * relevance
                    - (1 - self.lambda_param) * redundancy
                )

                if score > best_score:
                    best_score = score
                    best_index = index

            selected_indices.append(best_index)
            remaining_indices.remove(best_index)

            # Save the actual MMR score for this selection.
            mmr_scores[best_index] = float(best_score)

        selected_candidates = []

        for index in selected_indices:
            candidate = dict(candidates[index])

            candidate["mmr_score"] = mmr_scores[index]

            selected_candidates.append(candidate)

        return selected_candidates