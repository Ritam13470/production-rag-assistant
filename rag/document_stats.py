import os

import chromadb

from rag.config import DATA_DIR, DB_DIR, COLLECTION_NAME


def get_file_size_kb(file_path):
    size_bytes = os.path.getsize(file_path)
    return round(size_bytes / 1024, 2)


def list_source_files():
    if not os.path.exists(DATA_DIR):
        return []

    files = []

    for file_name in os.listdir(DATA_DIR):
        lower_name = file_name.lower()

        if lower_name.endswith((".txt", ".pdf")):
            file_path = os.path.join(DATA_DIR, file_name)

            files.append(
                {
                    "name": file_name,
                    "path": file_path,
                    "size_kb": get_file_size_kb(file_path),
                    "type": lower_name.split(".")[-1]
                }
            )

    return files


def get_total_file_size_kb(files):
    total_size = 0

    for file_info in files:
        total_size += file_info["size_kb"]

    return round(total_size, 2)


def chroma_database_exists():
    return os.path.exists(DB_DIR) and bool(os.listdir(DB_DIR))


def get_chroma_chunk_count():
    if not chroma_database_exists():
        return 0

    try:
        client = chromadb.PersistentClient(
            path=DB_DIR
        )

        collection = client.get_collection(
            name=COLLECTION_NAME
        )

        return collection.count()

    except Exception:
        return 0


def get_document_dashboard_stats():
    files = list_source_files()

    return {
        "file_count": len(files),
        "files": files,
        "total_size_kb": get_total_file_size_kb(files),
        "chroma_exists": chroma_database_exists(),
        "chunk_count": get_chroma_chunk_count()
    }
