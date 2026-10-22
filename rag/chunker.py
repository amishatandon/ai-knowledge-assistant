import re


class TextChunker:

    def __init__(
        self,
        chunk_size=1000,
        chunk_overlap=200
    ):

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller "
                "than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Try to preserve natural semantic boundaries
        self.separators = [
            "\n\n",   # paragraphs
            "\n",     # lines
            ". ",     # sentences
            "? ",
            "! ",
            " ",      # words
            ""        # characters (last resort)
        ]

    def _split_text(
        self,
        text,
        separator
    ):

        if separator == "":
            return list(text)

        return [
            part.strip()
            for part in text.split(separator)
            if part.strip()
        ]

    def _recursive_split(
        self,
        text,
        separators=None
    ):

        if separators is None:
            separators = self.separators

        text = text.strip()

        if not text:
            return []

        # If text already fits, keep it intact
        if len(text) <= self.chunk_size:
            return [text]

        # No separators left → character fallback
        if not separators:
            return [
                text[i:i + self.chunk_size]
                for i in range(
                    0,
                    len(text),
                    self.chunk_size
                )
            ]

        separator = separators[0]

        parts = self._split_text(
            text,
            separator
        )

        # Current separator wasn't useful
        if len(parts) <= 1:

            return self._recursive_split(
                text,
                separators[1:]
            )

        chunks = []
        current = ""

        for part in parts:

            # If an individual part is too large,
            # recursively split it using a smaller separator.
            if len(part) > self.chunk_size:

                if current:
                    chunks.append(
                        current.strip()
                    )
                    current = ""

                smaller_chunks = (
                    self._recursive_split(
                        part,
                        separators[1:]
                    )
                )

                chunks.extend(
                    smaller_chunks
                )

                continue

            if not current:

                candidate = part

            else:

                candidate = (
                    current
                    + separator
                    + part
                )

            if len(candidate) <= self.chunk_size:

                current = candidate

            else:

                if current:
                    chunks.append(
                        current.strip()
                    )

                current = part

        if current:
            chunks.append(
                current.strip()
            )

        return chunks

    def _add_overlap(
        self,
        chunks
    ):

        if not chunks:
            return []

        final_chunks = []

        for i, chunk in enumerate(chunks):

            if i == 0:

                final_chunks.append(
                    chunk
                )

                continue

            previous_chunk = chunks[i - 1]

            # Get whole words from previous chunk
            words = previous_chunk.split()

            overlap_words = []
            overlap_length = 0

            # Take words from the END of the
            # previous chunk.
            for word in reversed(words):

                additional_length = (
                    len(word)
                    + (
                        1
                        if overlap_words
                        else 0
                    )
                )

                if (
                    overlap_length
                    + additional_length
                    > self.chunk_overlap
                ):
                    break

                overlap_words.insert(
                    0,
                    word
                )

                overlap_length += (
                    additional_length
                )

            overlap_text = " ".join(
                overlap_words
            )

            if overlap_text:

                combined = (
                    overlap_text
                    + " "
                    + chunk
                )

                # Make absolutely sure the
                # resulting chunk stays within
                # the configured size.
                if len(combined) <= self.chunk_size:

                    final_chunks.append(
                        combined
                    )

                else:

                    final_chunks.append(
                        chunk
                    )

            else:

                final_chunks.append(
                    chunk
                )

        return final_chunks

    def chunk_documents(
        self,
        documents
    ):

        chunks = []

        for document in documents:

            text = document["text"]

            metadata = document["metadata"]

            # Step 1:
            # Recursively split using natural
            # boundaries.
            base_chunks = (
                self._recursive_split(
                    text
                )
            )

            # Step 2:
            # Add word-safe overlap.
            final_chunks = (
                self._add_overlap(
                    base_chunks
                )
            )

            # Step 3:
            # Add metadata.
            for chunk in final_chunks:

                if not chunk:
                    continue

                chunks.append({

                    "text": chunk,

                    "metadata": {
                        **metadata,
                        "chunk_id": len(chunks)
                    }

                })

        return chunks