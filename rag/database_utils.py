import os
import shutil

from chromadb.api.shared_system_client import SharedSystemClient

from rag.config import DATA_DIR, DB_DIR


def reset_chroma_client_cache():
    SharedSystemClient.clear_system_cache()


def clear_vector_database(db_dir=DB_DIR):
    reset_chroma_client_cache()

    normalized_db_dir = os.path.normpath(db_dir)
    absolute_db_dir = os.path.abspath(normalized_db_dir)
    absolute_data_dir = os.path.abspath(DATA_DIR)
    current_directory = os.path.abspath(os.getcwd())

    unsafe_targets = {
        "",
        ".",
        os.path.abspath("."),
        current_directory,
        absolute_data_dir
    }

    if absolute_db_dir in unsafe_targets:
        raise ValueError(
            "Refusing to delete an unsafe folder. Only the generated vector database folder can be cleared."
        )

    if not os.path.exists(normalized_db_dir):
        return {
            "deleted": False,
            "message": "Vector database folder does not exist."
        }

    shutil.rmtree(normalized_db_dir)
    reset_chroma_client_cache()

    return {
        "deleted": True,
        "message": f"Vector database folder cleared: {normalized_db_dir}"
    }
