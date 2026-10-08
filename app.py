from agent.rag_agent import RAGAgent


def main():

    print("=" * 70)
    print("AI KNOWLEDGE ASSISTANT")
    print("=" * 70)

    print(
        "\nKnowledge base loaded."
    )

    print(
        "Ask questions about your research papers."
    )

    print(
        "Type 'exit' to quit."
    )

    # -------------------------
    # Initialize RAG Agent
    # -------------------------

    try:

        agent = RAGAgent()

    except Exception as e:

        print(
            "\nERROR while initializing the agent:"
        )

        print(e)

        return

    # -------------------------
    # Question loop
    # -------------------------

    while True:

        print("\n" + "-" * 70)

        query = input(
            "\nAsk a question: "
        ).strip()

        # -------------------------
        # Exit
        # -------------------------

        if query.lower() == "exit":

            print("\nGoodbye!")

            break

        # -------------------------
        # Empty question
        # -------------------------

        if not query:

            print(
                "\nPlease enter a question."
            )

            continue

        # -------------------------
        # Generate answer
        # -------------------------

        try:

            result = agent.answer(
                query,
                top_k=5
            )

        except Exception as e:

            print(
                "\nERROR while answering the question:"
            )

            print(e)

            continue

        # -------------------------
        # Display answer
        # -------------------------

        print("\n" + "=" * 70)

        print("ANSWER")

        print("=" * 70)

        print(
            result["answer"]
        )

        # -------------------------
        # Display sources
        # -------------------------

        print("\n" + "=" * 70)

        print("SOURCES")

        print("=" * 70)

        results = result["results"]

        documents = (
            results["documents"][0]
        )

        metadatas = (
            results["metadatas"][0]
        )

        rrf_scores = (
            results["rrf_scores"][0]
        )

        reranker_scores = (
            results["reranker_scores"][0]
        )

        # -------------------------
        # Display each source
        # -------------------------

        for i in range(
            len(documents)
        ):

            metadata = metadatas[i]

            print(
                f"\n[{i + 1}] "
                f"{metadata['source']}"
            )

            print(
                f"    Page: "
                f"{metadata['page']}"
            )

            print(
                f"    Chunk: "
                f"{metadata['chunk_id']}"
            )

            print(
                f"    RRF Score: "
                f"{rrf_scores[i]:.6f}"
            )

            print(
                f"    Reranker Score: "
                f"{reranker_scores[i]:.4f}"
            )


# -------------------------
# Program Entry Point
# -------------------------

if __name__ == "__main__":

    main()
