from rag.retriever import Retriever


QUESTION = "Why is BERT different from earlier language models?"


def main():
    # Initialize the pipeline once.
    # Reuse it for both retrieval configurations.
    retriever = Retriever(query_rewriting=True)

    unique_chunks = {}

    # Inspect retrieval results from both configurations.
    for enabled in [False, True]:

        retriever.query_rewriting = enabled

        mode = (
            "WITH QUERY REWRITING"
            if enabled
            else "WITHOUT QUERY REWRITING"
        )

        print("\n" + "=" * 75)
        print(mode)
        print("=" * 75)

        results = retriever.retrieve(
            QUESTION,
            top_k=5
        )

        documents = results["documents"][0]
        ids = results["ids"][0]
        metadatas = results["metadatas"][0]

        for doc_id, document, metadata in zip(
            ids, documents, metadatas
        ):
            if doc_id not in unique_chunks:
                unique_chunks[doc_id] = {
                    "document": document,
                    "metadata": metadata,
                }

    # Display the unique retrieved chunks and their actual text.
    print("\n\n" + "#" * 75)
    print("INSPECT THE RETRIEVED CHUNKS BELOW")
    print("#" * 75)

    for index, (doc_id, item) in enumerate(
        unique_chunks.items(),
        start=1
    ):
        metadata = item["metadata"]
        document = item["document"]

        print("\n" + "-" * 75)
        print(f"Chunk {index}")
        print(f"ID: {doc_id}")
        print(f"Source: {metadata.get('source', 'Unknown')}")
        print(f"Page: {metadata.get('page', 'N/A')}")
        print("-" * 75)
        print(document)
        print()


if __name__ == "__main__":
    main()
