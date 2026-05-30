from io import BytesIO
import os
import py_compile
import shutil
from pathlib import Path

from rag.database_utils import clear_vector_database
from rag.document_manager import delete_source_file
from rag.document_stats import get_document_dashboard_stats
from rag.history import (
    add_query_history_entry,
    clear_query_history,
    get_query_history,
    get_query_history_count,
    initialize_query_history
)
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


def main():
    print("Starting no-API smoke test...")
    print("=" * 60)

    compile_python_files()
    test_safe_upload_helpers()
    test_document_delete_helper()
    test_vector_database_clear_helper()
    test_document_dashboard_stats()
    test_query_history_helpers()

    print("=" * 60)
    print("Smoke test passed successfully.")
    print("No Gemini API calls were made.")


if __name__ == "__main__":
    main()
