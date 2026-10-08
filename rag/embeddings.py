from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(
        self,
        model_name="all-MiniLM-L6-v2"
    ):

        print(
            f"Loading embedding model: {model_name}"
        )

        self.model = SentenceTransformer(
            model_name
        )

    def embed_text(self, text):

        embedding = self.model.encode(
            text,
            convert_to_numpy=True
        )

        return embedding

    def embed_documents(self, documents):

        texts = [
            document["text"]
            for document in documents
        ]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        return embeddings