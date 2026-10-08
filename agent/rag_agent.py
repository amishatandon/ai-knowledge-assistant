from agent.llm import GeminiLLM
from rag.retriever import Retriever
from rag.context import ContextBuilder


class RAGAgent:

    def __init__(self):

        self.retriever = Retriever()

        self.context_builder = (
            ContextBuilder()
        )

        self.llm = GeminiLLM()

    def answer(
        self,
        query,
        top_k=5
    ):

        # -------------------------
        # 1. Retrieve
        # -------------------------

        results = self.retriever.retrieve(
            query,
            top_k=top_k
        )

        # -------------------------
        # 2. Build context
        # -------------------------

        context = (
            self.context_builder.build_context(
                results
            )
        )

        # -------------------------
        # 3. Build prompt
        # -------------------------

        prompt = f"""
You are a research knowledge assistant.

Your job is to answer the user's question
using ONLY the retrieved context provided below.

IMPORTANT RULES:

1. Do not use outside knowledge.

2. Do not invent facts.

3. Every factual claim must have a citation.

4. Use the citation IDs provided in the context.

5. Place citations immediately after the
   claim they support.

6. If multiple sources support a claim,
   cite all relevant sources.

7. If the context does not contain enough
   information to answer the question,
   clearly say that the available documents
   do not contain enough information.

8. Never create a citation ID that does
   not exist in the context.

9. Do not cite a source merely because it
   is related to the topic. Cite it only
   when the retrieved content actually
   supports the claim.

CITATION FORMAT:

Use:

[1]

or:

[1][3]

Example:

The Transformer architecture uses
attention mechanisms rather than recurrence. [1]

The two approaches differ in how they
represent contextual information. [1][2]

-------------------------
RETRIEVED CONTEXT
-------------------------

{context}

-------------------------
USER QUESTION
-------------------------

{query}

-------------------------
ANSWER
-------------------------
"""

        # -------------------------
        # 4. Generate answer
        # -------------------------

        answer = self.llm.generate(
            prompt
        )

        # -------------------------
        # 5. Return answer
        # -------------------------

        return {
            "answer": answer,
            "results": results
        }
