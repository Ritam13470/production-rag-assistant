import os
import subprocess
import sys

import streamlit as st
from dotenv import load_dotenv

from rag.config import DB_DIR
from rag.database_utils import clear_vector_database
from rag.document_manager import delete_source_file
from rag.document_stats import get_document_dashboard_stats
from rag.errors import RagPipelineError
from rag.pipeline import answer_question
from rag.upload_utils import save_uploaded_files
from rag.utils import preview_text, format_source_label

load_dotenv()


def rebuild_vector_database():
    result = subprocess.run(
        [sys.executable, "ingest.py"],
        capture_output=True,
        text=True
    )

    return result


def initialize_query_history():
    if "query_history" not in st.session_state:
        st.session_state.query_history = []


def add_query_history_entry(question, answer, scored_docs):
    sources = []

    for doc, score in scored_docs:
        sources.append(
            {
                "source": doc.metadata.get("source", "Unknown source"),
                "page": doc.metadata.get("page"),
                "type": doc.metadata.get("type"),
                "score": score,
                "preview": preview_text(doc.page_content, max_chars=500)
            }
        )

    st.session_state.query_history.insert(
        0,
        {
            "question": question,
            "answer": answer,
            "sources": sources
        }
    )


st.set_page_config(
    page_title="Production RAG Assistant",
    page_icon="??",
    layout="wide"
)

initialize_query_history()

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

    st.header("Session History")

    st.metric("Questions asked", len(st.session_state.query_history))

    if st.button("Clear Query History"):
        st.session_state.query_history = []
        st.success("Query history cleared.")

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
        try:
            saved_files = save_uploaded_files(uploaded_files)

            st.success("Files saved successfully.")

            for file_info in saved_files:
                st.write(
                    f"{file_info['original_name']} ? {file_info['saved_name']}"
                )

            st.info("Refresh the page to update the sidebar file list.")

        except ValueError as error:
            st.error(str(error))

st.subheader("2. Delete Uploaded Document")

if stats["files"]:
    file_names = [
        file_info["name"]
        for file_info in stats["files"]
    ]

    selected_file_name = st.selectbox(
        "Choose a document to delete",
        file_names
    )

    st.warning(
        "Deleting a document removes it from the data folder only. "
        "After deleting, rebuild the vector database so old chunks are removed from ChromaDB."
    )

    confirm_delete = st.checkbox(
        f"I understand and want to delete {selected_file_name}"
    )

    if st.button("Delete Selected Document"):
        if not confirm_delete:
            st.warning("Please tick the confirmation checkbox before deleting the document.")
        else:
            try:
                delete_result = delete_source_file(selected_file_name)

                if delete_result["deleted"]:
                    st.success(delete_result["message"])
                    st.warning("Now rebuild the vector database to remove old chunks for this document.")
                else:
                    st.info(delete_result["message"])

                st.info("Refresh the page to update the sidebar document list.")

            except ValueError as error:
                st.error(str(error))
else:
    st.info("No uploaded TXT/PDF documents are available to delete.")

st.subheader("3. Manage Vector Database")

st.write("Rebuild the vector database after adding, deleting, or changing files.")

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

st.warning("Clearing the vector database deletes only generated ChromaDB files. Your uploaded documents stay inside the data folder.")

confirm_clear = st.checkbox(
    "I understand and want to enable clearing the vector database"
)

if st.button("Clear Vector Database"):
    if not confirm_clear:
        st.warning("Please tick the confirmation checkbox before clearing the vector database.")
    else:
        try:
            clear_result = clear_vector_database()

            if clear_result["deleted"]:
                st.success(clear_result["message"])
            else:
                st.info(clear_result["message"])

            st.info("Refresh the page to update the sidebar dashboard stats.")

        except ValueError as error:
            st.error(str(error))

st.subheader("4. Ask Questions")

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

            add_query_history_entry(
                question=question,
                answer=result.answer,
                scored_docs=result.scored_docs
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

st.subheader("5. Query History")

if st.session_state.query_history:
    for index, history_item in enumerate(st.session_state.query_history, start=1):
        with st.expander(f"{index}. {history_item['question']}"):
            st.markdown("**Answer**")
            st.write(history_item["answer"])

            st.markdown("**Sources used**")

            for source_index, source in enumerate(history_item["sources"], start=1):
                page = source.get("page")
                page_text = f", page {page}" if page else ""

                st.write(
                    f"{source_index}. {source['source']}{page_text} "
                    f"(distance: {source['score']:.4f})"
                )
                st.caption(source["preview"])
else:
    st.info("No questions asked yet in this session.")
