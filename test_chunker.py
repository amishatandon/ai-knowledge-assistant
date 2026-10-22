from rag.chunker import TextChunker


def main():

    text = """
    The Transformer is a neural network architecture
    based on attention mechanisms.

    Unlike recurrent neural networks, the Transformer
    can process sequence elements in parallel.

    The architecture consists of encoder and decoder
    components.

    Multi-head attention allows the model to jointly
    attend to information from different representation
    subspaces.
    """

    documents = [
        {
            "text": text.strip(),
            "metadata": {
                "source": "test.pdf",
                "page": 1
            }
        }
    ]

    chunker = TextChunker(
        chunk_size=200,
        chunk_overlap=50
    )

    chunks = chunker.chunk_documents(
        documents
    )

    print(
        f"Total chunks: {len(chunks)}"
    )

    print("\n" + "=" * 70)

    for chunk in chunks:

        print(
            f"\nChunk ID: "
            f"{chunk['metadata']['chunk_id']}"
        )

        print(
            f"Source: "
            f"{chunk['metadata']['source']}"
        )

        print(
            f"Page: "
            f"{chunk['metadata']['page']}"
        )

        print(
            f"Length: "
            f"{len(chunk['text'])}"
        )

        print(
            f"Text:\n"
            f"{chunk['text']}"
        )

        print("-" * 70)


if __name__ == "__main__":
    main()