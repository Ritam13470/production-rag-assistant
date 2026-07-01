import importlib
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


REQUIRED_FILES = [
    "app.py",
    "ask.py",
    "ingest.py",
    "evaluate.py",
    "smoke_test.py",
    "README.md",
    "requirements.txt",
    "eval_questions.json"
]

REQUIRED_FOLDERS = [
    "rag",
    "data"
]

REQUIRED_RAG_FILES = [
    "rag/__init__.py",
    "rag/config.py",
    "rag/database_utils.py",
    "rag/document_manager.py",
    "rag/document_stats.py",
    "rag/embeddings.py",
    "rag/errors.py",
    "rag/history.py",
    "rag/observability.py",
    "rag/pipeline.py",
    "rag/prompts.py",
    "rag/providers.py",
    "rag/upload_utils.py",
    "rag/utils.py"
]

REQUIRED_IMPORTS = [
    "streamlit",
    "dotenv",
    "pypdf",
    "chromadb",
    "langchain_chroma",
    "langchain_core",
    "langchain_google_genai",
    "langchain_groq",
    "langchain_text_splitters"
]


def check_python_version():
    print("Checking Python version...")

    major = sys.version_info.major
    minor = sys.version_info.minor

    print(f"Python version: {major}.{minor}.{sys.version_info.micro}")

    if major < 3 or (major == 3 and minor < 10):
        raise RuntimeError("Python 3.10 or newer is recommended for this project.")

    print("OK: Python version")


def check_paths(paths, label):
    print(f"Checking {label}...")

    missing = []

    for item in paths:
        path = Path(item)

        if not path.exists():
            missing.append(item)
        else:
            print(f"OK: {item}")

    if missing:
        raise FileNotFoundError(
            f"Missing {label}: {missing}"
        )


def is_real_secret(value, placeholders):
    if value is None:
        return False

    normalized = value.strip().lower()

    return normalized not in placeholders


def check_env_file():
    print("Checking environment file...")

    env_path = Path(".env")

    if not env_path.exists():
        raise FileNotFoundError(
            ".env file is missing. Create it from .env.example or add GOOGLE_API_KEY manually."
        )

    load_dotenv()

    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not is_real_secret(
        google_api_key,
        {"your_api_key_here", "replace_me", ""}
    ):
        raise RuntimeError(
            "GOOGLE_API_KEY is missing or looks like a placeholder."
        )

    chat_provider = os.getenv("CHAT_PROVIDER", "gemini").strip().lower() or "gemini"

    if chat_provider not in {"gemini", "groq"}:
        raise RuntimeError(
            "CHAT_PROVIDER must be either gemini or groq."
        )

    if chat_provider == "groq":
        groq_api_key = os.getenv("GROQ_API_KEY")

        if not is_real_secret(
            groq_api_key,
            {"your_groq_api_key_here", "replace_me", ""}
        ):
            raise RuntimeError(
                "GROQ_API_KEY is required when CHAT_PROVIDER=groq."
            )

        print("OK: GROQ_API_KEY appears configured")

    print("OK: .env found")
    print("OK: GOOGLE_API_KEY appears configured")
    print(f"OK: CHAT_PROVIDER is {chat_provider}")


def check_imports():
    print("Checking required package imports...")

    failed_imports = []

    for module_name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
            print(f"OK: {module_name}")
        except Exception as error:
            failed_imports.append(
                f"{module_name}: {error}"
            )

    if failed_imports:
        raise ImportError(
            "Some required packages could not be imported:\n"
            + "\n".join(failed_imports)
        )


def main():
    print("Starting startup checklist...")
    print("=" * 60)

    check_python_version()
    check_paths(REQUIRED_FILES, "required files")
    check_paths(REQUIRED_FOLDERS, "required folders")
    check_paths(REQUIRED_RAG_FILES, "required rag files")
    check_env_file()
    check_imports()

    print("=" * 60)
    print("Startup checklist passed successfully.")
    print("No Gemini API calls were made.")


if __name__ == "__main__":
    main()
