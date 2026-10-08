import re

from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, vector_store):

        self.vector_store = vector_store

        self.documents = []
        self.metadatas = []
        self.ids = []

        self.bm25 = None

        self._build_index()

    def _tokenize(self, text):

        text = text.lower()

        tokens = re.findall(
            r"\b\w+\b",
            text
        )

        return tokens

    def _build_index(self):

        results = self.vector_store.collection.get(
            include=[
                "documents",
                "metadatas"
            ]
        )

        self.documents = (
            results.get("documents") or []
        )

        self.metadatas = (
            results.get("metadatas") or []
        )

        self.ids = (
            results.get("ids") or []
        )

        if not self.documents:

            print(
                "WARNING: No documents found "
                "for BM25 index."
            )

            return

        tokenized_documents = [
            self._tokenize(document)
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

        print(
            f"BM25 index built with "
            f"{len(self.documents)} chunks."
        )

    def search(
        self,
        query,
        top_k=10
    ):

        if self.bm25 is None:

            return []

        query_tokens = self._tokenize(
            query
        )

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []

        for index in ranked_indices[:top_k]:

            results.append({
                "id": self.ids[index],
                "document": self.documents[index],
                "metadata": self.metadatas[index],
                "score": float(scores[index]),
                "rank": len(results) + 1
            })

        return results
