import os
import re

from rag.config import DATA_DIR

ALLOWED_EXTENSIONS = {".txt", ".pdf"}


def sanitize_file_name(original_name):
    safe_base_name = original_name.replace("\\", "/").split("/")[-1]
    safe_base_name = safe_base_name.strip()

    name_part, extension = os.path.splitext(safe_base_name)
    extension = extension.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}. Only TXT and PDF files are allowed."
        )

    clean_name = re.sub(r"[^A-Za-z0-9_-]+", "_", name_part)
    clean_name = clean_name.strip("_")

    if not clean_name:
        clean_name = "document"

    clean_name = clean_name[:80]

    return f"{clean_name}{extension}"


def get_unique_file_path(file_name, data_dir=DATA_DIR):
    os.makedirs(data_dir, exist_ok=True)

    name_part, extension = os.path.splitext(file_name)
    file_path = os.path.join(data_dir, file_name)

    counter = 2

    while os.path.exists(file_path):
        file_path = os.path.join(
            data_dir,
            f"{name_part}_{counter}{extension}"
        )
        counter += 1

    return file_path


def save_uploaded_files(uploaded_files, data_dir=DATA_DIR):
    saved_files = []

    for uploaded_file in uploaded_files:
        safe_file_name = sanitize_file_name(uploaded_file.name)
        file_path = get_unique_file_path(
            file_name=safe_file_name,
            data_dir=data_dir
        )

        with open(file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())

        saved_files.append(
            {
                "original_name": uploaded_file.name,
                "saved_name": os.path.basename(file_path),
                "path": file_path
            }
        )

    return saved_files
