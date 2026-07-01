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


def check_env_file():
    print("Checking environment file...")

    env_path = Path(".env")

    if not env_path.exists():
        raise FileNotFoundError(
            ".env file is missing. Create it from .env.example or add GOOGLE_API_KEY manually."
        )

    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing from environment variables."
        )

    if api_key.strip().lower() in ["your_api_key_here", "replace_me", ""]:
        raise RuntimeError(
            "GOOGLE_API_KEY looks like a placeholder. Replace it with a real key."
        )

    print("OK: .env found")
    print("OK: GOOGLE_API_KEY appears configured")


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
