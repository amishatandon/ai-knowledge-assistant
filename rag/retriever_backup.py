
from rag.embeddings import EmbeddingModel
from rag.vectorstore import VectorStore
from rag.bm25 import BM25Retriever
from rag.reranker import Reranker
from rag.mmr import MMR
from agent.llm import GeminiLLM


print(">>> LOADED UPDATED rag/retriever.py WITH QUERY REWRITING <<<")
print(">>> FILE PATH:", __file__)


class Retriever:

    def __init__(self):

        # Embeddings and vector database
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()

        # BM25 keyword retrieval
        self.bm25_retriever = BM25Retriever(
            self.vector_store
        )

        # MMR diversity selection
        self.mmr = MMR(
            self.embedding_model,
            lambda_param=0.7
        )

        # Cross-Encoder reranking
        self.reranker = Reranker()

        # LLM for query rewriting
        self.query_rewriter = GeminiLLM()

    # --------------------------------------------------
    # Stage 1: Query Rewriting
    # --------------------------------------------------

    def _rewrite_query(self, query):

        print("\n===== QUERY REWRITING =====")
        print("Original query:", query)

        prompt = f"""
You are a search-query optimization component
for a research-paper knowledge assistant.

Rewrite the user's question into one clear,
standalone search query that is suitable for
retrieving relevant passages from research papers.

Rules:
1. Preserve the user's original intent.
2. Do not answer the question.
3. Do not introduce unsupported facts or assumptions.
4. Preserve important technical terms, names, and acronyms.
5. Remove unnecessary conversational wording.
6. If the query is already clear, keep it nearly unchanged.
7. Return only the rewritten query, without explanations,
   quotation marks, or labels.

User question:
{query}

Rewritten search query:
"""

        try:

            rewritten_query = self.query_rewriter.generate(
                prompt
            )

            if not isinstance(rewritten_query, str):
                print(
                    "Query rewriting returned an invalid result."
                )
                print("Using original query.")
                return query

            rewritten_query = rewritten_query.strip()

            # Remove surrounding quotation marks, if present.
            rewritten_query = rewritten_query.strip("\"'").strip()

            # Fall back if the LLM returns an empty response.
            if not rewritten_query:
                print(
                    "Empty rewritten query. Using original query."
                )
                return query

            print("Rewritten query:", rewritten_query)
            print("============================\n")

            return rewritten_query

        except Exception as e:

            print(
                "Query rewriting failed. "
                "Falling back to the original query."
            )
            print("Reason:", e)

            return query

    # --------------------------------------------------
    # Stage 2: Dense Semantic Search
    # --------------------------------------------------

    def _dense_search(self, query, top_k=10):

        query_embedding = self.embedding_model.embed_text(
            query
        )

        results = self.vector_store.search(
            query_embedding,
            top_k=top_k
        )

        dense_results = []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]

        for i in range(len(documents)):

            dense_results.append({
                "id": ids[i],
                "document": documents[i],
                "metadata": metadatas[i],
                "distance": distances[i],
                "rank": i + 1
            })

        return dense_results

    # --------------------------------------------------
    # Stage 3: Reciprocal Rank Fusion
    # --------------------------------------------------

    def _rrf_fusion(
        self,
        dense_results,
        bm25_results,
        k=60
    ):

        fused_scores = {}
        result_data = {}

        # Add dense retrieval rankings.
        for result in dense_results:

            document_id = result["id"]
            rank = result["rank"]

            score = 1 / (k + rank)

            fused_scores[document_id] = (
                fused_scores.get(document_id, 0) + score
            )

            result_data[document_id] = result

        # Add BM25 keyword rankings.
        for result in bm25_results:

            document_id = result["id"]
            rank = result["rank"]

            score = 1 / (k + rank)

            fused_scores[document_id] = (
                fused_scores.get(document_id, 0) + score
            )

            if document_id not in result_data:
                result_data[document_id] = result

        # Sort by combined RRF score.
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

            result = result_data[document_id]

            fused_results.append({
                "id": document_id,
                "document": result["document"],
                "metadata": result["metadata"],
                "rrf_score": fused_scores[document_id],
                "rank": rank
            })

        return fused_results

    # --------------------------------------------------
    # Stage 4: Complete Retrieval Pipeline
    # --------------------------------------------------

    def retrieve(self, query, top_k=5):

        print("\n>>> Retriever.retrieve() WAS CALLED <<<")

        # Preserve the user's original question.
        original_query = query

        # Stage 1: Rewrite the query for retrieval.
        search_query = self._rewrite_query(
            original_query
        )

        # Number of candidates retrieved from each retriever.
        candidate_k = max(top_k * 4, 20)

        # Stage 2: Dense semantic retrieval.
        print(">>> Running dense retrieval <<<")

        dense_results = self._dense_search(
            search_query,
            top_k=candidate_k
        )

        # Stage 3: BM25 keyword retrieval.
        print(">>> Running BM25 retrieval <<<")

        bm25_results = self.bm25_retriever.search(
            search_query,
            top_k=candidate_k
        )

        # Stage 4: Combine dense and BM25 rankings.
        print(">>> Running RRF fusion <<<")

        fused_results = self._rrf_fusion(
            dense_results,
            bm25_results
        )

        # Stage 5: Select relevant, diverse chunks.
        print(">>> Running MMR selection <<<")

        candidates = fused_results[:candidate_k]

        mmr_results = self.mmr.select(
            search_query,
            candidates,
            top_k=min(10, len(candidates))
        )

        # Stage 6: Rerank using the Cross-Encoder.
        print(">>> Running Cross-Encoder reranking <<<")

        reranked_results = self.reranker.rerank(
            search_query,
            mmr_results,
            top_k=top_k
        )

        # Stage 7: Debug scores.
        print("\n===== RETRIEVAL SCORE DEBUG =====")
        print("Original query:", original_query)
        print("Search query:", search_query)

        for result in reranked_results:

            print(
                f"ID: {result['id']}\n"
                f"RRF Score: "
                f"{result.get('rrf_score', 'MISSING')}\n"
                f"MMR Score: "
                f"{result.get('mmr_score', 'MISSING')}\n"
                f"Reranker Score: "
                f"{result.get('reranker_score', 'MISSING')}\n"
            )

        print("=================================\n")

        # Stage 8: Return results in the format expected
        # by ContextBuilder and the existing app.py.
        return {
            "original_query": original_query,
            "rewritten_query": search_query,

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

            "mmr_scores": [[
                result["mmr_score"]
                for result in reranked_results
            ]],

            "reranker_scores": [[
                result["reranker_score"]
                for result in reranked_results
            ]]
        }