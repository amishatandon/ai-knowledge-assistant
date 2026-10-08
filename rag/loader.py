from pathlib import Path
from pypdf import PdfReader


class DocumentLoader:

    def load_pdf(self, file_path):
        """
        Extract text from a PDF file.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of dictionaries containing page text and metadata.
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        reader = PdfReader(str(file_path))

        documents = []

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text()

            if text:
                documents.append({
                    "text": text.strip(),
                    "metadata": {
                        "source": file_path.name,
                        "page": page_number
                    }
                })

        return documents