import os

from dotenv import load_dotenv


load_dotenv()


DATA_DIR = "data"
DB_DIR = "chroma_db"
COLLECTION_NAME = "rag_documents"

DEFAULT_CHAT_PROVIDER = "gemini"
DEFAULT_EMBEDDING_PROVIDER = "gemini"

SUPPORTED_CHAT_PROVIDERS = {"gemini"}
SUPPORTED_EMBEDDING_PROVIDERS = {"gemini"}

DEFAULT_GEMINI_EMBEDDING_MODEL = "gemini-embedding-2-preview"
DEFAULT_GEMINI_CHAT_MODEL = "gemini-2.5-flash"


def read_config_value(name, default):
    value = os.getenv(name, default)

    if value is None:
        return default

    value = value.strip()

    if not value:
        return default

    return value


def read_provider_name(name, default, supported_providers):
    provider = read_config_value(name, default).lower()

    if provider not in supported_providers:
        supported = ", ".join(sorted(supported_providers))
        raise ValueError(
            f"Unsupported {name}: {provider}. "
            f"Supported values: {supported}."
        )

    return provider


CHAT_PROVIDER = read_provider_name(
    "CHAT_PROVIDER",
    DEFAULT_CHAT_PROVIDER,
    SUPPORTED_CHAT_PROVIDERS
)

EMBEDDING_PROVIDER = read_provider_name(
    "EMBEDDING_PROVIDER",
    DEFAULT_EMBEDDING_PROVIDER,
    SUPPORTED_EMBEDDING_PROVIDERS
)

EMBEDDING_MODEL = read_config_value(
    "GEMINI_EMBEDDING_MODEL",
    DEFAULT_GEMINI_EMBEDDING_MODEL
)

CHAT_MODEL = read_config_value(
    "GEMINI_CHAT_MODEL",
    DEFAULT_GEMINI_CHAT_MODEL
)

MAX_INGEST_CHUNKS = 100
