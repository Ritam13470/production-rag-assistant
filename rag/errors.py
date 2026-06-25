class RagPipelineError(Exception):
    def __init__(self, user_message, technical_details=""):
        super().__init__(user_message)

        self.user_message = user_message
        self.technical_details = technical_details


def create_rag_error(error):
    error_text = str(error)
    lower_error_text = error_text.lower()

    if (
        "document_chunk_limit_exceeded" in lower_error_text
        or "chunk safety limit" in lower_error_text
    ):
        return RagPipelineError(
            user_message=(
                "The uploaded documents are too large for this learning/demo deployment. "
                "Please use fewer or smaller documents, then rebuild the vector database."
            ),
            technical_details=error_text
        )

    if (
        "resource_exhausted" in lower_error_text
        or "quota" in lower_error_text
        or "rate limit" in lower_error_text
        or "429" in lower_error_text
    ):
        return RagPipelineError(
            user_message=(
                "Gemini API quota or rate limit was reached. "
                "Please wait for a while and try again."
            ),
            technical_details=error_text
        )

    if (
        "api key" in lower_error_text
        or "api_key" in lower_error_text
        or "unauthorized" in lower_error_text
        or "permission" in lower_error_text
        or "403" in lower_error_text
        or "401" in lower_error_text
    ):
        return RagPipelineError(
            user_message=(
                "The Gemini API key is missing, invalid, or not allowed to use this model. "
                "Please check your .env file and API key settings."
            ),
            technical_details=error_text
        )

    if (
        "chroma" in lower_error_text
        or "collection" in lower_error_text
        or "no such file" in lower_error_text
        or "does not exist" in lower_error_text
    ):
        return RagPipelineError(
            user_message=(
                "The vector database could not be loaded. "
                "Please rebuild the vector database first."
            ),
            technical_details=error_text
        )

    return RagPipelineError(
        user_message=(
            "The RAG pipeline failed unexpectedly. "
            "Please check the technical details and try again."
        ),
        technical_details=error_text
    )
