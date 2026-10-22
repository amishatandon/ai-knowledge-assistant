from agent.llm import GeminiLLM
from rag.embeddings import EmbeddingModel
from rag.vectorstore import VectorStore
from rag.bm25 import BM25Retriever
from rag.reranker import Reranker
from rag.mmr import MMR


class Retriever:

    def __init__(self, query_rewriting=True):
        """
        Initialize the retrieval pipeline.

        Args:
            query_rewriting (bool):
                True  -> Rewrite the user's query using Gemini.
                False -> Use the original query directly.
        """

        print("\nInitializing Retriever...")

        self.query_rewriting = query_rewriting

        # Stage 1: Embedding model
        self.embedding_model = EmbeddingModel()

        # Stage 2: Vector database
        self.vector_store = VectorStore()

        # Stage 3: Keyword retriever
        self.bm25_retriever = BM25Retriever(
            self.vector_store
        )

        # Stage 4: MMR diversity selection
        self.mmr = MMR(
            self.embedding_model,
            lambda_param=0.7
        )

        # Stage 5: Cross-Encoder reranking
        self.reranker = Reranker()

        # Initialize Gemini only when query rewriting is enabled.
        self.query_rewriter = (
            GeminiLLM() if self.query_rewriting else None
        )

        print(
            "Query rewriting:",
            "ENABLED" if self.query_rewriting else "DISABLED"
        )

        print("Retriever initialized successfully.")

    def _rewrite_query(self, query):
        """
        Rewrite the user's question into a search-friendly query.

        If rewriting fails, fall back to the original query.
        """

        print("\n===== QUERY REWRITING =====")
        print("Original query:", query)

        if not self.query_rewriting:
            print("Query rewriting is disabled.")
            return query

        prompt = f"""
You are an expert search-query optimization system for a
research-paper knowledge assistant.

Rewrite the user's question into a concise search query that
helps retrieve relevant passages from research papers.

Rules:
1. Preserve the original meaning and intent.
2. Keep important technical terms, model names, and acronyms.
3. Add relevant terminology only when it is clearly implied.
4. Do not answer the question.
5. Return only the rewritten search query.
6. Do not add explanations, labels, or quotation marks.

User question:
{query}

Rewritten search query:
"""

        try:
            rewritten_query = self.query_rewriter.generate(
                prompt
            )

            if not rewritten_query:
                print("Empty rewrite. Using original query.")
                return query

            rewritten_query = rewritten_query.strip()

            if not rewritten_query:
                print("Empty rewrite. Using original query.")
                return query

            print("Rewritten query:", rewritten_query)

            return rewritten_query

        except Exception as e:
            print("Query rewriting failed:", str(e))
            print("Falling back to the original query.")

            return query

    def _dense_search(self, query, top_k=10):
        """Retrieve candidates using semantic similarity."""

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

    def _rrf_fusion(
        self,
        dense_results,
        bm25_results,
        k=60
    ):
        """
        Combine dense and BM25 rankings using
        Reciprocal Rank Fusion (RRF).
        """

        fused_scores = {}
        result_data = {}

        # Stage 1: Add dense retrieval scores.
        for result in dense_results:
            document_id = result["id"]
            rank = result["rank"]

            score = 1 / (k + rank)

            fused_scores[document_id] = (
                fused_scores.get(document_id, 0) + score
            )

            result_data[document_id] = result

        # Stage 2: Add BM25 keyword retrieval scores.
        for result in bm25_results:
            document_id = result["id"]
            rank = result["rank"]

            score = 1 / (k + rank)

            fused_scores[document_id] = (
                fused_scores.get(document_id, 0) + score
            )

            if document_id not in result_data:
                result_data[document_id] = result

        # Stage 3: Rank documents by their fused scores.
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

    def retrieve(self, query, top_k=5):
        """
        Run the complete retrieval pipeline.

        The query_rewriting setting determines whether the
        original query or rewritten query is searched.
        """

        original_query = query.strip()

        if not original_query:
            raise ValueError("Query cannot be empty.")

        print("\n" + "=" * 60)
        print("RETRIEVAL PIPELINE")
        print("=" * 60)

        # Stage 0: Optionally rewrite the query.
        search_query = self._rewrite_query(
            original_query
        )

        print("\nOriginal query:", original_query)
        print("Search query:", search_query)

        # Retrieve more candidates than the final requested
        # number, allowing subsequent stages to refine them.
        candidate_k = max(top_k * 4, 20)

        # Stage 1: Dense semantic retrieval.
        print("\n>>> Running dense retrieval <<<")

        dense_results = self._dense_search(
            search_query,
            top_k=candidate_k
        )

        # Stage 2: BM25 keyword retrieval.
        print("\n>>> Running BM25 retrieval <<<")

        bm25_results = self.bm25_retriever.search(
            search_query,
            top_k=candidate_k
        )

        # Stage 3: Reciprocal Rank Fusion.
        print("\n>>> Running RRF fusion <<<")

        fused_results = self._rrf_fusion(
            dense_results,
            bm25_results
        )

        # Stage 4: MMR selects relevant and diverse chunks.
        print("\n>>> Running MMR selection <<<")

        candidates = fused_results[:candidate_k]

        mmr_results = self.mmr.select(
            search_query,
            candidates,
            top_k=min(10, len(candidates))
        )

        # Stage 5: Cross-Encoder reranking.
        print("\n>>> Running Cross-Encoder reranking <<<")

        reranked_results = self.reranker.rerank(
            search_query,
            mmr_results,
            top_k=top_k
        )

        # Debugging: inspect retrieval scores.
        print("\n===== RETRIEVAL SCORE DEBUG =====")

        print("Original query:", original_query)
        print("Search query:", search_query)

        for result in reranked_results:
            print(
                f"\nID: {result['id']}\n"
                f"RRF Score: "
                f"{result.get('rrf_score', 'MISSING')}\n"
                f"MMR Score: "
                f"{result.get('mmr_score', 'MISSING')}\n"
                f"Reranker Score: "
                f"{result.get('reranker_score', 'MISSING')}\n"
            )

        print("=================================\n")

        # Return the same data structure expected by the
        # existing ContextBuilder and app.py.
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
