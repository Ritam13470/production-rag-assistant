import os
import subprocess
import sys

import streamlit as st
from dotenv import load_dotenv

from rag.config import DATA_DIR, DB_DIR
from rag.document_stats import get_document_dashboard_stats
from rag.errors import RagPipelineError
from rag.pipeline import answer_question
from rag.utils import preview_text, format_source_label

load_dotenv()


def save_uploaded_files(uploaded_files):
    os.makedirs(DATA_DIR, exist_ok=True)

    saved_files = []

    for uploaded_file in uploaded_files:
        file_path = os.path.join(DATA_DIR, uploaded_file.name)

        with open(file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())

        saved_files.append(file_path)

    return saved_files


def rebuild_vector_database():
    result = subprocess.run(
        [sys.executable, "ingest.py"],
        capture_output=True,
        text=True
    )

    return result


st.set_page_config(
    page_title="Production RAG Assistant",
    page_icon="??",
    layout="wide"
)

st.title("Production RAG Assistant")
st.write("Upload TXT/PDF files, rebuild the vector database, and ask questions using Gemini, LangChain, and ChromaDB.")

stats = get_document_dashboard_stats()

with st.sidebar:
    st.header("Project Info")
    st.write("Vector DB: ChromaDB")
    st.write("Chat Model: Gemini 2.5 Flash")
    st.write("Embedding Model: Gemini Embedding")
    st.warning("For learning, upload sample documents only. Avoid private or sensitive files.")

    st.divider()

    st.header("Document Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Files", stats["file_count"])

    with col2:
        st.metric("Chunks", stats["chunk_count"])

    st.metric("Total file size", f"{stats['total_size_kb']} KB")

    if stats["chroma_exists"]:
        st.success("Vector database found")
    else:
        st.error("Vector database not found")

    st.divider()

    st.header("Retrieval Settings")
    top_k = st.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=8,
        value=3
    )

    show_debug = st.checkbox(
        "Show retrieval debug panel",
        value=True
    )

    st.caption("Chroma distance score: lower usually means more similar.")

    st.divider()

    st.header("Current Files")

    if stats["files"]:
        for file_info in stats["files"]:
            st.write(
                f"- {file_info['name']} "
                f"({file_info['type']}, {file_info['size_kb']} KB)"
            )
    else:
        st.write("No TXT/PDF files found.")

st.subheader("1. Upload Documents")

uploaded_files = st.file_uploader(
    "Upload TXT or PDF files",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Save Uploaded Files"):
        saved_files = save_uploaded_files(uploaded_files)

        st.success("Files saved successfully.")

        for file_path in saved_files:
            st.write(file_path)

st.subheader("2. Rebuild Vector Database")

st.write("Click this after adding or changing files.")

if st.button("Rebuild Vector Database"):
    with st.spinner("Rebuilding vector database. This may take a moment..."):
        result = rebuild_vector_database()

    if result.returncode == 0:
        st.success("Vector database rebuilt successfully.")
        st.code(result.stdout)
        st.info("Refresh the page to update the sidebar dashboard stats.")
    else:
        st.error("Vector database rebuild failed.")
        st.code(result.stdout)
        st.code(result.stderr)

st.subheader("3. Ask Questions")

question = st.text_input(
    "Ask a question",
    placeholder="Example: What backend technologies does CharacterForge AI use?"
)

if st.button("Ask"):
    if not question.strip():
        st.warning("Please type a question first.")
    elif not os.path.exists(DB_DIR):
        st.error("Vector database not found. Please click Rebuild Vector Database first.")
    else:
        try:
            with st.spinner("Searching documents and generating answer..."):
                result = answer_question(
                    question=question,
                    top_k=top_k
                )

            st.subheader("Answer")
            st.write(result.answer)

            st.subheader("Sources")

            if result.scored_docs:
                for index, item in enumerate(result.scored_docs, start=1):
                    doc, score = item

                    with st.expander(f"Source {index}: {format_source_label(doc, score)}"):
                        st.write(preview_text(doc.page_content))
            else:
                st.write("No sources found.")

            if show_debug:
                st.subheader("Retrieval Debug Panel")

                st.write("This panel shows the chunks retrieved before Gemini generated the answer.")

                for index, item in enumerate(result.scored_docs, start=1):
                    doc, score = item

                    st.markdown(f"**Chunk {index}**")
                    st.write(format_source_label(doc, score))
                    st.code(preview_text(doc.page_content, max_chars=1500))

        except RagPipelineError as error:
            st.error(error.user_message)

            with st.expander("Technical details"):
                st.code(error.technical_details)
