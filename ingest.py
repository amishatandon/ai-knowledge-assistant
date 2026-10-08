from pathlib import Path
import hashlib

from rag.loader import DocumentLoader
from rag.chunker import TextChunker
from rag.embeddings import EmbeddingModel
from rag.vectorstore import VectorStore


# ============================================================
# CONFIGURATION
# ============================================================

DOCUMENTS_DIR = "documents"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

VECTOR_DB_PATH = "chroma_db"
COLLECTION_NAME = "knowledge_base"


# ============================================================
# FILE HASH
# ============================================================

def calculate_file_hash(
    file_path
):

    sha256 = hashlib.sha256()

    with open(
        file_path,
        "rb"
    ) as file:

        while True:

            data = file.read(
                1024 * 1024
            )

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


# ============================================================
# INGESTION
# ============================================================

def main():

    print("=" * 70)
    print("AI KNOWLEDGE AGENT - DOCUMENT INGESTION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Locate documents folder
    # --------------------------------------------------------

    documents_path = Path(
        DOCUMENTS_DIR
    )

    if not documents_path.exists():

        print(
            f"\nERROR: '{DOCUMENTS_DIR}' "
            f"folder does not exist."
        )

        return

    # --------------------------------------------------------
    # 2. Find PDFs
    # --------------------------------------------------------

    pdf_files = list(
        documents_path.glob("*.pdf")
    )

    if not pdf_files:

        print(
            "\nNo PDF files found."
        )

        return

    print(
        f"\nFound {len(pdf_files)} PDF file(s)."
    )

    # --------------------------------------------------------
    # 3. Initialize components
    # --------------------------------------------------------

    loader = DocumentLoader()

    chunker = TextChunker(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    vector_store = VectorStore(
        path=VECTOR_DB_PATH,
        collection_name=COLLECTION_NAME
    )

    # --------------------------------------------------------
    # 4. Check which documents need processing
    # --------------------------------------------------------

    documents_to_process = []

    skipped_documents = []

    reindexed_documents = []

    for pdf_file in pdf_files:

        print("\n" + "-" * 70)

        print(
            f"Checking: {pdf_file.name}"
        )

        # ----------------------------------------------------
        # Calculate current file hash
        # ----------------------------------------------------

        current_hash = (
            calculate_file_hash(
                pdf_file
            )
        )

        print(
            f"Hash: {current_hash[:16]}..."
        )

        # ----------------------------------------------------
        # Check existing document
        # ----------------------------------------------------

        existing_hash = (
            vector_store.get_document_hash(
                pdf_file.name
            )
        )

        # ----------------------------------------------------
        # CASE 1: Document doesn't exist
        # ----------------------------------------------------

        if existing_hash is None:

            if vector_store.document_exists(
                pdf_file.name
            ):

                print(
                    "Existing document found "
                    "without hash metadata."
                )

                print(
                    "It will be re-indexed."
                )

                vector_store.delete_document(
                    pdf_file.name
                )

            else:

                print(
                    "New document detected."
                )

            documents_to_process.append(
                (
                    pdf_file,
                    current_hash,
                    "new"
                )
            )

        # ----------------------------------------------------
        # CASE 2: Document unchanged
        # ----------------------------------------------------

        elif existing_hash == current_hash:

            print(
                "Document unchanged."
            )

            print(
                "Skipping ingestion."
            )

            skipped_documents.append(
                pdf_file.name
            )

        # ----------------------------------------------------
        # CASE 3: Document changed
        # ----------------------------------------------------

        else:

            print(
                "Document has changed."
            )

            print(
                "Removing old version..."
            )

            vector_store.delete_document(
                pdf_file.name
            )

            documents_to_process.append(
                (
                    pdf_file,
                    current_hash,
                    "updated"
                )
            )

            reindexed_documents.append(
                pdf_file.name
            )

    # --------------------------------------------------------
    # 5. Nothing to process
    # --------------------------------------------------------

    if not documents_to_process:

        print("\n" + "=" * 70)
        print("NO NEW DOCUMENTS TO INDEX")
        print("=" * 70)

        print(
            "\nAll documents are already up to date."
        )

        print(
            f"\nDocuments skipped: "
            f"{len(skipped_documents)}"
        )

        print(
            f"Total vectors in ChromaDB: "
            f"{vector_store.count()}"
        )

        return

    # --------------------------------------------------------
    # 6. Initialize embedding model
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOADING EMBEDDING MODEL")
    print("=" * 70)

    embedding_model = EmbeddingModel()

    total_pages = 0
    total_chunks = 0
    total_documents = 0

    # --------------------------------------------------------
    # 7. Process required documents
    # --------------------------------------------------------

    for (
        pdf_file,
        document_hash,
        status
    ) in documents_to_process:

        print("\n" + "=" * 70)

        if status == "new":

            print(
                f"INDEXING NEW DOCUMENT: "
                f"{pdf_file.name}"
            )

        else:

            print(
                f"RE-INDEXING DOCUMENT: "
                f"{pdf_file.name}"
            )

        print("=" * 70)

        # ----------------------------------------------------
        # Load PDF
        # ----------------------------------------------------

        documents = loader.load_pdf(
            pdf_file
        )

        print(
            f"\nPages loaded: "
            f"{len(documents)}"
        )

        if not documents:

            print(
                "WARNING: No text extracted."
            )

            continue

        total_pages += len(
            documents
        )

        # ----------------------------------------------------
        # Add document hash to metadata
        # ----------------------------------------------------

        for document in documents:

            document["metadata"][
                "document_hash"
            ] = document_hash

        # ----------------------------------------------------
        # Chunk document
        # ----------------------------------------------------

        chunks = chunker.chunk_documents(
            documents
        )

        print(
            f"Chunks created: "
            f"{len(chunks)}"
        )

        total_chunks += len(
            chunks
        )

        # ----------------------------------------------------
        # Generate embeddings
        # ----------------------------------------------------

        print(
            "\nGenerating embeddings..."
        )

        embeddings = (
            embedding_model.embed_documents(
                chunks
            )
        )

        print(
            f"Embedding shape: "
            f"{embeddings.shape}"
        )

        # ----------------------------------------------------
        # Store in ChromaDB
        # ----------------------------------------------------

        print(
            "\nStoring in ChromaDB..."
        )

        vector_store.add_documents(
            chunks,
            embeddings
        )

        total_documents += 1

        print(
            f"Successfully indexed: "
            f"{pdf_file.name}"
        )

    # --------------------------------------------------------
    # 8. Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)

    print(
        f"\nNew/updated PDFs indexed : "
        f"{total_documents}"
    )

    print(
        f"PDFs skipped             : "
        f"{len(skipped_documents)}"
    )

    print(
        f"Pages processed          : "
        f"{total_pages}"
    )

    print(
        f"Chunks created           : "
        f"{total_chunks}"
    )

    print(
        f"Vectors in ChromaDB      : "
        f"{vector_store.count()}"
    )

    print(
        "\nKnowledge base is up to date."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()