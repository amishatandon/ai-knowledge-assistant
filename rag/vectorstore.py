import chromadb


class VectorStore:

    def __init__(
        self,
        path="chroma_db",
        collection_name="knowledge_base"
    ):

        self.client = chromadb.PersistentClient(
            path=path
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": "cosine"
                }
            )
        )

    # ========================================================
    # ADD DOCUMENTS
    # ========================================================

    def add_documents(
        self,
        documents,
        embeddings
    ):

        ids = []

        for document in documents:

            metadata = document["metadata"]

            source = metadata["source"]
            page = metadata["page"]
            chunk_id = metadata["chunk_id"]

            document_id = (
                f"{source}-"
                f"page-{page}-"
                f"chunk-{chunk_id}"
            )

            ids.append(document_id)

        self.collection.add(

            ids=ids,

            documents=[
                document["text"]
                for document in documents
            ],

            embeddings=embeddings.tolist(),

            metadatas=[
                document["metadata"]
                for document in documents
            ]
        )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query_embedding,
        top_k=5
    ):

        results = self.collection.query(

            query_embeddings=[
                query_embedding.tolist()
            ],

            n_results=top_k
        )

        return results

    # ========================================================
    # COUNT
    # ========================================================

    def count(self):

        return self.collection.count()

    # ========================================================
    # CHECK DOCUMENT
    # ========================================================

    def get_document_hash(
        self,
        source
    ):

        results = self.collection.get(

            where={
                "source": source
            },

            include=[
                "metadatas"
            ]
        )

        metadatas = results.get(
            "metadatas"
        )

        if not metadatas:

            return None

        # All chunks of the same document
        # should contain the same document_hash.

        first_metadata = metadatas[0]

        return first_metadata.get(
            "document_hash"
        )

    # ========================================================
    # DELETE DOCUMENT
    # ========================================================

    def delete_document(
        self,
        source
    ):

        self.collection.delete(

            where={
                "source": source
            }
        )

    # ========================================================
    # DOCUMENT EXISTS
    # ========================================================

    def document_exists(
        self,
        source
    ):

        results = self.collection.get(

            where={
                "source": source
            },

            include=[
                "metadatas"
            ]
        )

        metadatas = results.get(
            "metadatas"
        )

        return bool(metadatas)