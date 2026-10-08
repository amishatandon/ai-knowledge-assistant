from rag.embeddings import EmbeddingModel
from rag.vectorstore import VectorStore
from rag.bm25 import BM25Retriever
from rag.reranker import Reranker


class Retriever:

    def __init__(self):

        self.embedding_model = (
            EmbeddingModel()
        )

        self.vector_store = (
            VectorStore()
        )

        self.bm25_retriever = (
            BM25Retriever(
                self.vector_store
            )
        )

        self.reranker = (
            Reranker()
        )

    # --------------------------------------------------
    # Dense Retrieval
    # --------------------------------------------------

    def _dense_search(
        self,
        query,
        top_k=10
    ):

        query_embedding = (
            self.embedding_model.embed_text(
                query
            )
        )

        results = self.vector_store.search(
            query_embedding,
            top_k=top_k
        )

        dense_results = []

        documents = results[
            "documents"
        ][0]

        metadatas = results[
            "metadatas"
        ][0]

        distances = results[
            "distances"
        ][0]

        ids = results[
            "ids"
        ][0]

        for i in range(
            len(documents)
        ):

            dense_results.append({

                "id": ids[i],

                "document": documents[i],

                "metadata": metadatas[i],

                "distance": distances[i],

                "rank": i + 1
            })

        return dense_results

    # --------------------------------------------------
    # Reciprocal Rank Fusion
    # --------------------------------------------------

    def _rrf_fusion(
        self,
        dense_results,
        bm25_results,
        k=60
    ):

        fused_scores = {}

        result_data = {}

        # -------------------------
        # Dense results
        # -------------------------

        for result in dense_results:

            document_id = result[
                "id"
            ]

            rank = result[
                "rank"
            ]

            score = 1 / (
                k + rank
            )

            fused_scores[
                document_id
            ] = fused_scores.get(
                document_id,
                0
            ) + score

            result_data[
                document_id
            ] = result

        # -------------------------
        # BM25 results
        # -------------------------

        for result in bm25_results:

            document_id = result[
                "id"
            ]

            rank = result[
                "rank"
            ]

            score = 1 / (
                k + rank
            )

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

        # -------------------------
        # Sort by RRF score
        # -------------------------

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

    # --------------------------------------------------
    # Hybrid Retrieval + Reranking
    # --------------------------------------------------

    def retrieve(
        self,
        query,
        top_k=5
    ):

        # Retrieve more candidates than
        # we finally return.

        candidate_k = max(
            top_k * 4,
            20
        )

        # -------------------------
        # 1. Dense retrieval
        # -------------------------

        dense_results = (
            self._dense_search(
                query,
                top_k=candidate_k
            )
        )

        # -------------------------
        # 2. BM25 retrieval
        # -------------------------

        bm25_results = (
            self.bm25_retriever.search(
                query,
                top_k=candidate_k
            )
        )

        # -------------------------
        # 3. RRF fusion
        # -------------------------

        fused_results = (
            self._rrf_fusion(
                dense_results,
                bm25_results
            )
        )

        # -------------------------
        # 4. Reranking
        # -------------------------

        reranked_results = (
            self.reranker.rerank(
                query,
                fused_results[:candidate_k],
                top_k=top_k
            )
        )

        # -------------------------
        # 5. Convert to
        #    existing RAG format
        # -------------------------

        return {

            "documents": [[
                result["document"]
                for result in reranked_results
            ]],

            "metadatas": [[
                result["metadata"]
                for result in reranked_results
            ]],

            "ids": [[
                result["id"]
                for result in reranked_results
            ]],

            "rrf_scores": [[
                result["rrf_score"]
                for result in reranked_results
            ]],

            "reranker_scores": [[
                result["reranker_score"]
                for result in reranked_results
            ]]
        }
