import os
import shutil
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DATA_DIR = "data"
DB_DIR = "chroma_db"
COLLECTION_NAME = "rag_documents"
EMBEDDING_MODEL = "gemini-embedding-2-preview"


class SafeGoogleEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model=model_name
        )

    def embed_documents(self, texts):
        vectors = []

        for index, text in enumerate(texts, start=1):
            print(f"Creating embedding for chunk {index} of {len(texts)}...")
            vector = self.embedding_model.embed_query(text)
            vectors.append(vector)

        return vectors

    def embed_query(self, text):
        return self.embedding_model.embed_query(text)


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

    print(f"Created chunks: {len(chunks)}")

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
