from langchain_google_genai import ChatGoogleGenerativeAI

from rag.config import (
    CHAT_MODEL,
    CHAT_PROVIDER,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER
)
from rag.embeddings import SafeGoogleEmbeddings


def get_provider_status():
    return {
        "chat_provider": CHAT_PROVIDER,
        "embedding_provider": EMBEDDING_PROVIDER,
        "chat_model": CHAT_MODEL,
        "embedding_model": EMBEDDING_MODEL
    }


def build_embedding_model():
    if EMBEDDING_PROVIDER == "gemini":
        return SafeGoogleEmbeddings(
            model_name=EMBEDDING_MODEL
        )

    raise ValueError(
        f"Unsupported embedding provider: {EMBEDDING_PROVIDER}"
    )


def build_chat_model():
    if CHAT_PROVIDER == "gemini":
        return ChatGoogleGenerativeAI(
            model=CHAT_MODEL,
            temperature=0
        )

    raise ValueError(
        f"Unsupported chat provider: {CHAT_PROVIDER}"
    )
