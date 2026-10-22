from agent.llm import GeminiLLM


class QueryRewriter:

    def __init__(self, llm=None):

        self.llm = llm or GeminiLLM()

    def rewrite(self, query):

        query = query.strip()

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        prompt = f"""
You are a query rewriting component for a
research-paper Retrieval-Augmented Generation system.

Your task is to rewrite the user's question into
one clear, specific query suitable for document retrieval.

Rules:
1. Preserve the original intent.
2. Do not answer the question.
3. Do not invent facts, entities, papers, or assumptions.
4. Preserve important technical terms and named entities.
5. Resolve ambiguity only when the original question
   provides enough information.
6. If the question is already clear, keep it mostly unchanged.
7. Return only the rewritten query, without explanations,
   labels, quotation marks, or markdown.

Original question:
{query}

Rewritten query:
"""

        try:
            rewritten_query = self.llm.generate(prompt)

            if not rewritten_query:
                return query

            rewritten_query = rewritten_query.strip()

            # Remove surrounding quotation marks if present.
            rewritten_query = rewritten_query.strip(
                "\"' "
            )

            # Fall back to the original query if the model
            # returns an empty result.
            if not rewritten_query:
                return query

            return rewritten_query

        except Exception as error:
            print(
                "Query rewriting failed; "
                f"using original query. Details: {error}"
            )

            return query
