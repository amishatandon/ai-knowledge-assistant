class TextChunker:

    def __init__(self, chunk_size=1000, chunk_overlap=200):

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, documents):

        chunks = []

        for document in documents:

            text = document["text"]
            metadata = document["metadata"]

            start = 0

            while start < len(text):

                end = start + self.chunk_size

                chunk_text = text[start:end].strip()

                if chunk_text:

                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            **metadata,
                            "chunk_id": len(chunks)
                        }
                    })

                start = end - self.chunk_overlap

        return chunks