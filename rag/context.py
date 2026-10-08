class ContextBuilder:

    def build_context(self, results):

        documents = results[
            "documents"
        ][0]

        metadatas = results[
            "metadatas"
        ][0]

        rrf_scores = results[
            "rrf_scores"
        ][0]

        reranker_scores = results[
            "reranker_scores"
        ][0]

        context_parts = []

        for i in range(
            len(documents)
        ):

            text = documents[i]

            metadata = metadatas[i]

            rrf_score = rrf_scores[i]

            reranker_score = (
                reranker_scores[i]
            )

            citation_id = i + 1

            source = metadata[
                "source"
            ]

            page = metadata[
                "page"
            ]

            chunk_id = metadata[
                "chunk_id"
            ]

            context_part = f"""
[{citation_id}]

SOURCE: {source}
PAGE: {page}
CHUNK: {chunk_id}

RRF SCORE: {rrf_score:.6f}
RERANKER SCORE: {reranker_score:.4f}

CONTENT:
{text}
"""

            context_parts.append(
                context_part.strip()
            )

        return "\n\n".join(
            context_parts
        )
