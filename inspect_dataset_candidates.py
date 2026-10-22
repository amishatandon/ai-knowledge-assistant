from rag.retriever import Retriever


QUESTIONS = [
    {
        "question": (
            "How does self-attention help Transformers "
            "model relationships between tokens?"
        ),
        "topic": "Transformer"
    },
    {
        "question": (
            "How does LoRA reduce the number of trainable "
            "parameters during fine-tuning?"
        ),
        "topic": "LoRA"
    },
    {
        "question": (
            "How does Retrieval-Augmented Generation "
            "combine retrieval with text generation?"
        ),
        "topic": "RAG"
    },
    {
        "question": (
            "How does the ReAct framework combine reasoning "
            "and actions?"
        ),
        "topic": "ReAct"
    },
    {
        "question": (
            "How does Dense Passage Retrieval use encoders "
            "to retrieve relevant passages?"
        ),
        "topic": "DPR"
    }
]


def main():
    # No Gemini client is needed for query rewriting.
    retriever = Retriever(query_rewriting=False)

    for item in QUESTIONS:
        question = item["question"]

        print("\n" + "=" * 85)
        print(f"TOPIC: {item['topic']}")
        print(f"QUESTION: {question}")
        print("=" * 85)

        results = retriever.retrieve(
            question,
            top_k=5
        )

        documents = results["documents"][0]
        ids = results["ids"][0]
        metadatas = results["metadatas"][0]

        for rank, (doc_id, document, metadata) in enumerate(
            zip(ids, documents, metadatas),
            start=1
        ):
            print("\n" + "-" * 75)
            print(f"Rank: {rank}")
            print(f"ID: {doc_id}")
            print(f"Source: {metadata.get('source', 'Unknown')}")
            print(f"Page: {metadata.get('page', 'N/A')}")
            print("\nChunk text:")
            print(document[:1200])
            print("-" * 75)


if __name__ == "__main__":
    main()