from io import BytesIO
import os
import py_compile
import shutil
import subprocess
import sys
from pathlib import Path

from rag.config import (
    SUPPORTED_CHAT_PROVIDERS,
    SUPPORTED_EMBEDDING_PROVIDERS,
    read_provider_name
)
from rag.database_utils import clear_vector_database
from rag.errors import create_rag_error
from rag.document_manager import delete_source_file
from rag.document_stats import get_document_dashboard_stats
from rag.history import (
    add_query_history_entry,
    clear_query_history,
    get_query_history,
    get_query_history_count,
    initialize_query_history
)
from rag.observability import get_langsmith_status
from rag.providers import get_provider_status
from rag.upload_utils import sanitize_file_name, save_uploaded_files


class FakeUploadedFile:
    def __init__(self, name, content):
        self.name = name
        self.content = BytesIO(content)

    def getbuffer(self):
        return self.content.getbuffer()


class FakeDocument:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


class FakeSessionState:
    def __contains__(self, key):
        return hasattr(self, key)


def compile_python_files():
    print("Checking Python syntax...")

    paths = [
        Path("app.py"),
        Path("ask.py"),
        Path("ingest.py"),
        Path("evaluate.py")
    ]

    paths.extend(sorted(Path("rag").glob("*.py")))

    for path in paths:
        py_compile.compile(str(path), doraise=True)
        print(f"OK: {path}")


def test_safe_upload_helpers():
    print("Testing safe upload helpers...")

    assert sanitize_file_name("../../my file (final)!!.PDF") == "my_file_final.pdf"
    assert sanitize_file_name("notes.txt") == "notes.txt"

    test_dir = "tmp_smoke_upload"

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    files = [
        FakeUploadedFile("../../my file (final)!!.PDF", b"pdf test"),
        FakeUploadedFile("my file (final)!!.PDF", b"pdf duplicate"),
        FakeUploadedFile("notes.txt", b"txt test")
    ]

    saved_files = save_uploaded_files(
        uploaded_files=files,
        data_dir=test_dir
    )

    saved_names = [
        file_info["saved_name"]
        for file_info in saved_files
    ]

    assert saved_names == [
        "my_file_final.pdf",
        "my_file_final_2.pdf",
        "notes.txt"
    ]

    shutil.rmtree(test_dir)
    print("OK: safe upload helpers")


def test_document_delete_helper():
    print("Testing document delete helper...")

    test_dir = "tmp_smoke_delete"

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    os.makedirs(test_dir, exist_ok=True)

    test_file = os.path.join(test_dir, "delete_me.txt")

    with open(test_file, "w", encoding="utf-8") as file:
        file.write("temporary text")

    result = delete_source_file(
        file_name="delete_me.txt",
        data_dir=test_dir
    )

    assert result["deleted"] is True
    assert not os.path.exists(test_file)

    try:
        delete_source_file(
            file_name="../secret.txt",
            data_dir=test_dir
        )
        raise AssertionError("Path-style filename was not rejected.")
    except ValueError:
        pass

    shutil.rmtree(test_dir)
    print("OK: document delete helper")


def test_vector_database_clear_helper():
    print("Testing vector database clear helper...")

    test_dir = "tmp_smoke_chroma"

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    os.makedirs(test_dir, exist_ok=True)

    with open(os.path.join(test_dir, "sample.txt"), "w", encoding="utf-8") as file:
        file.write("temporary vector db file")

    result = clear_vector_database(
        db_dir=test_dir
    )

    assert result["deleted"] is True
    assert not os.path.exists(test_dir)

    print("OK: vector database clear helper")


def test_document_dashboard_stats():
    print("Testing document dashboard stats...")

    stats = get_document_dashboard_stats()

    required_keys = {
        "file_count",
        "files",
        "total_size_kb",
        "chroma_exists",
        "chunk_count"
    }

    assert required_keys.issubset(stats.keys())

    print("OK: document dashboard stats")


def test_query_history_helpers():
    print("Testing query history helpers...")

    session_state = FakeSessionState()

    initialize_query_history(session_state)

    assert get_query_history_count(session_state) == 0

    doc = FakeDocument(
        page_content="This is a short retrieved chunk preview for smoke testing.",
        metadata={
            "source": "data/test.txt",
            "type": "txt"
        }
    )

    add_query_history_entry(
        session_state=session_state,
        question="What is this?",
        answer="This is a smoke test answer.",
        scored_docs=[(doc, 0.1234)]
    )

    assert get_query_history_count(session_state) == 1

    history = get_query_history(session_state)

    assert history[0]["question"] == "What is this?"
    assert history[0]["answer"] == "This is a smoke test answer."
    assert history[0]["sources"][0]["source"] == "data/test.txt"

    clear_query_history(session_state)

    assert get_query_history_count(session_state) == 0

    print("OK: query history helpers")


def test_rag_error_classification():
    print("Testing RAG error classification...")

    large_document_error = create_rag_error(
        RuntimeError(
            "DOCUMENT_CHUNK_LIMIT_EXCEEDED: "
            "The uploaded documents created 200 chunks, "
            "which exceeds the configured chunk safety limit."
        )
    )

    assert "too large" in large_document_error.user_message
    assert "DOCUMENT_CHUNK_LIMIT_EXCEEDED" in large_document_error.technical_details

    quota_error = create_rag_error(
        RuntimeError(
            "429 RESOURCE_EXHAUSTED: Gemini API quota was reached."
        )
    )

    assert "quota or rate limit" in quota_error.user_message
    assert "RESOURCE_EXHAUSTED" in quota_error.technical_details

    print("OK: RAG error classification")


def test_large_document_guard_preserves_database():
    print("Testing large document guard preserves database...")

    import ingest

    test_db_dir = "tmp_smoke_existing_chroma"

    if os.path.exists(test_db_dir):
        shutil.rmtree(test_db_dir)

    os.makedirs(test_db_dir, exist_ok=True)

    marker_file = os.path.join(test_db_dir, "existing_marker.txt")

    with open(marker_file, "w", encoding="utf-8") as file:
        file.write("existing database content")

    calls = {
        "embeddings": False,
        "chroma": False
    }

    original_load_documents = ingest.load_documents
    original_max_chunks = ingest.MAX_INGEST_CHUNKS
    original_db_dir = ingest.DB_DIR
    original_splitter = ingest.RecursiveCharacterTextSplitter
    original_embeddings = ingest.SafeGoogleEmbeddings
    original_chroma = ingest.Chroma

    class FakeSplitter:
        def __init__(self, chunk_size, chunk_overlap):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split_documents(self, documents):
            return [
                FakeDocument("chunk one", {"source": "data/large.txt"}),
                FakeDocument("chunk two", {"source": "data/large.txt"})
            ]

    class FailingEmbeddings:
        def __init__(self, model_name):
            calls["embeddings"] = True
            raise AssertionError("Embeddings should not be created when guard fails.")

    class FailingChroma:
        @staticmethod
        def from_documents(*args, **kwargs):
            calls["chroma"] = True
            raise AssertionError("Chroma should not be rebuilt when guard fails.")

    try:
        ingest.load_documents = lambda: [
            FakeDocument("large document", {"source": "data/large.txt"})
        ]
        ingest.MAX_INGEST_CHUNKS = 1
        ingest.DB_DIR = test_db_dir
        ingest.RecursiveCharacterTextSplitter = FakeSplitter
        ingest.SafeGoogleEmbeddings = FailingEmbeddings
        ingest.Chroma = FailingChroma

        try:
            ingest.main()
            raise AssertionError("Large document guard did not stop ingestion.")
        except RuntimeError as error:
            assert "DOCUMENT_CHUNK_LIMIT_EXCEEDED" in str(error)

        assert os.path.exists(marker_file)
        assert calls["embeddings"] is False
        assert calls["chroma"] is False

    finally:
        ingest.load_documents = original_load_documents
        ingest.MAX_INGEST_CHUNKS = original_max_chunks
        ingest.DB_DIR = original_db_dir
        ingest.RecursiveCharacterTextSplitter = original_splitter
        ingest.SafeGoogleEmbeddings = original_embeddings
        ingest.Chroma = original_chroma

        if os.path.exists(test_db_dir):
            shutil.rmtree(test_db_dir)

    print("OK: large document guard preserves database")




def test_langsmith_observability_defaults():
    print("Testing LangSmith observability defaults...")

    original_tracing = os.environ.get("LANGSMITH_TRACING")
    original_api_key = os.environ.get("LANGSMITH_API_KEY")
    original_project = os.environ.get("LANGSMITH_PROJECT")

    try:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGSMITH_API_KEY"] = ""
        os.environ["LANGSMITH_PROJECT"] = ""

        status = get_langsmith_status()

        assert status["enabled"] is False
        assert status["tracing_requested"] is False
        assert status["api_key_configured"] is False
        assert status["project_name"] == "production-rag-assistant"
        assert "disabled" in status["reason"].lower()

        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = ""

        status = get_langsmith_status()

        assert status["enabled"] is False
        assert status["tracing_requested"] is True
        assert status["api_key_configured"] is False
        assert "not configured" in status["reason"].lower()

        os.environ["LANGSMITH_API_KEY"] = "your_langsmith_api_key_here"

        status = get_langsmith_status()

        assert status["enabled"] is False
        assert status["api_key_configured"] is False

        os.environ["LANGSMITH_API_KEY"] = "lsv2_fake_test_key"

        status = get_langsmith_status()

        assert status["enabled"] is True
        assert status["api_key_configured"] is True
        assert status["project_name"] == "production-rag-assistant"

    finally:
        if original_tracing is None:
            os.environ.pop("LANGSMITH_TRACING", None)
        else:
            os.environ["LANGSMITH_TRACING"] = original_tracing

        if original_api_key is None:
            os.environ.pop("LANGSMITH_API_KEY", None)
        else:
            os.environ["LANGSMITH_API_KEY"] = original_api_key

        if original_project is None:
            os.environ.pop("LANGSMITH_PROJECT", None)
        else:
            os.environ["LANGSMITH_PROJECT"] = original_project

    print("OK: LangSmith observability defaults")



def test_provider_config_defaults():
    print("Testing provider configuration defaults...")

    original_test_provider = os.environ.get("TEST_PROVIDER")

    try:
        os.environ.pop("TEST_PROVIDER", None)

        assert "gemini" in SUPPORTED_CHAT_PROVIDERS
        assert "groq" in SUPPORTED_CHAT_PROVIDERS
        assert "gemini" in SUPPORTED_EMBEDDING_PROVIDERS
        assert "groq" not in SUPPORTED_EMBEDDING_PROVIDERS

        assert read_provider_name(
            "TEST_PROVIDER",
            "gemini",
            SUPPORTED_CHAT_PROVIDERS
        ) == "gemini"

        os.environ["TEST_PROVIDER"] = "GROQ"

        assert read_provider_name(
            "TEST_PROVIDER",
            "gemini",
            SUPPORTED_CHAT_PROVIDERS
        ) == "groq"

        os.environ["TEST_PROVIDER"] = "openai"

        try:
            read_provider_name(
                "TEST_PROVIDER",
                "gemini",
                SUPPORTED_CHAT_PROVIDERS
            )
            raise AssertionError("Unsupported provider was not rejected.")
        except ValueError as error:
            assert "Unsupported TEST_PROVIDER" in str(error)
            assert "gemini" in str(error)
            assert "groq" in str(error)

        status = get_provider_status()

        assert status["chat_provider"] in SUPPORTED_CHAT_PROVIDERS
        assert status["embedding_provider"] in SUPPORTED_EMBEDDING_PROVIDERS
        assert status["chat_model"]
        assert status["embedding_model"]

        env = os.environ.copy()
        env["CHAT_PROVIDER"] = "groq"
        env["EMBEDDING_PROVIDER"] = "gemini"
        env["GROQ_CHAT_MODEL"] = "openai/gpt-oss-20b"

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from rag.providers import get_provider_status; "
                    "status = get_provider_status(); "
                    "assert status['chat_provider'] == 'groq'; "
                    "assert status['embedding_provider'] == 'gemini'; "
                    "assert status['chat_model'] == 'openai/gpt-oss-20b'; "
                    "print(status)"
                )
            ],
            env=env,
            capture_output=True,
            text=True,
            check=True
        )

        assert "'chat_provider': 'groq'" in result.stdout

    finally:
        if original_test_provider is None:
            os.environ.pop("TEST_PROVIDER", None)
        else:
            os.environ["TEST_PROVIDER"] = original_test_provider

    print("OK: provider configuration defaults")


def main():
    print("Starting no-API smoke test...")
    print("=" * 60)

    compile_python_files()
    test_safe_upload_helpers()
    test_document_delete_helper()
    test_vector_database_clear_helper()
    test_document_dashboard_stats()
    test_query_history_helpers()
    test_rag_error_classification()
    test_large_document_guard_preserves_database()
    test_langsmith_observability_defaults()
    test_provider_config_defaults()

    print("=" * 60)
    print("Smoke test passed successfully.")
    print("No Gemini API calls were made.")


if __name__ == "__main__":
    main()
