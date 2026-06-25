import os
import shutil
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from rag.config import (
    DATA_DIR,
    DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    MAX_INGEST_CHUNKS
)
from rag.embeddings import SafeGoogleEmbeddings

load_dotenv()


def load_txt_file(file_path):
    print(f"Loading TXT file: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    return [
        Document(
            page_content=text,
            metadata={"source": file_path, "type": "txt"}
        )
    ]


def load_pdf_file(file_path):
    print(f"Loading PDF file: {file_path}")

    documents = []
    reader = PdfReader(file_path)

    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_path,
                        "type": "pdf",
                        "page": page_index
                    }
                )
            )

    return documents


def load_documents():
    documents = []

    for filename in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, filename)
        lower_name = filename.lower()

        if lower_name.endswith(".txt"):
            documents.extend(load_txt_file(file_path))

        elif lower_name.endswith(".pdf"):
            documents.extend(load_pdf_file(file_path))

    return documents


def main():
    print("Starting document ingestion...")

    documents = load_documents()

    if not documents:
        print("No .txt or .pdf files found inside the data folder.")
        return

    print(f"Loaded document sections/pages: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120
    )

    chunks = splitter.split_documents(documents)

    chunk_count = len(chunks)

    print(f"Created chunks: {chunk_count}")
    print(f"Configured chunk safety limit: {MAX_INGEST_CHUNKS}")

    if chunk_count > MAX_INGEST_CHUNKS:
        raise RuntimeError(
            "DOCUMENT_CHUNK_LIMIT_EXCEEDED: "
            f"The uploaded documents created {chunk_count} chunks, "
            f"which exceeds the configured safety limit of {MAX_INGEST_CHUNKS}. "
            "Use fewer or smaller documents and rebuild again."
        )

    if os.path.exists(DB_DIR):
        print("Removing old Chroma database...")
        shutil.rmtree(DB_DIR)

    embeddings = SafeGoogleEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name=COLLECTION_NAME
    )

    print("Chroma vector database created successfully.")
    print(f"Database folder: {DB_DIR}")


if __name__ == "__main__":
    main()
