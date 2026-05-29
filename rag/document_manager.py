import os

from rag.config import DATA_DIR

ALLOWED_EXTENSIONS = {".txt", ".pdf"}


def validate_file_name(file_name):
    stripped_name = file_name.strip()

    if not stripped_name:
        raise ValueError("File name cannot be empty.")

    if "/" in stripped_name or "\\" in stripped_name:
        raise ValueError(
            "Path-style filenames are not allowed. Only files directly inside the data folder can be deleted."
        )

    extension = os.path.splitext(stripped_name)[1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            "Only TXT and PDF files inside the data folder can be deleted."
        )

    return stripped_name


def get_safe_source_file_path(file_name, data_dir=DATA_DIR):
    safe_file_name = validate_file_name(file_name)

    data_root = os.path.abspath(data_dir)
    file_path = os.path.abspath(
        os.path.join(data_root, safe_file_name)
    )

    common_path = os.path.commonpath(
        [data_root, file_path]
    )

    if common_path != data_root:
        raise ValueError(
            "Refusing to delete a file outside the data folder."
        )

    return file_path


def delete_source_file(file_name, data_dir=DATA_DIR):
    file_path = get_safe_source_file_path(
        file_name=file_name,
        data_dir=data_dir
    )

    if not os.path.exists(file_path):
        return {
            "deleted": False,
            "message": f"File does not exist: {os.path.basename(file_path)}"
        }

    if not os.path.isfile(file_path):
        raise ValueError(
            "Refusing to delete because the target is not a file."
        )

    os.remove(file_path)

    return {
        "deleted": True,
        "message": f"Deleted file: {os.path.basename(file_path)}"
    }
