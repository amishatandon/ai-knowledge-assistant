import os
import time

from dotenv import load_dotenv
from google import genai


class GeminiLLM:

    def __init__(
        self,
        model="gemini-3.8-flash",
        max_retries=3,
        base_delay=2
    ):

        load_dotenv()

        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

        if not api_key:

            raise ValueError(
                "GEMINI_API_KEY not found "
                "in .env"
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.model = model

        self.max_retries = max_retries

        self.base_delay = base_delay

    def generate(self, prompt):

        for attempt in range(
            self.max_retries + 1
        ):

            try:

                response = (
                    self.client.models.generate_content(
                        model=self.model,
                        contents=prompt
                    )
                )

                return response.text

            except Exception as e:

                error_message = str(e)

                # ------------------------------------------------
                # Retry only transient server/service errors
                # ------------------------------------------------

                is_transient_error = (
                    "503" in error_message
                    or "UNAVAILABLE" in error_message
                    or "500" in error_message
                    or "INTERNAL" in error_message
                    or "502" in error_message
                    or "504" in error_message
                )

                if (
                    not is_transient_error
                    or attempt >= self.max_retries
                ):

                    raise e

                delay = (
                    self.base_delay
                    * (2 ** attempt)
                )

                print(
                    f"\nGemini temporarily unavailable."
                )

                print(
                    f"Retrying in {delay} seconds..."
                )

                print(
                    f"Attempt "
                    f"{attempt + 1}/"
                    f"{self.max_retries}"
                )

                time.sleep(delay)