import os
import subprocess
import sys

import streamlit as st
from dotenv import load_dotenv

from rag.config import DB_DIR
from rag.database_utils import clear_vector_database
from rag.document_manager import delete_source_file
from rag.document_stats import get_document_dashboard_stats
from rag.errors import RagPipelineError, create_rag_error
from rag.history import (
    add_query_history_entry,
    clear_query_history,
    get_query_history,
    get_query_history_count,
    initialize_query_history
)
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


def create_rebuild_error(result):
    error_text = "\n".join(
        part for part in [
            result.stdout or "",
            result.stderr or "",
            f"ingest.py exited with code {result.returncode}"
        ]
        if part.strip()
    )

    return create_rag_error(RuntimeError(error_text))


st.set_page_config(
    page_title="Production RAG Assistant",
    page_icon="??",
    layout="wide"
)

initialize_query_history(st.session_state)

stats = get_document_dashboard_stats()

st.title("Production RAG Assistant")
st.caption("A document-based RAG dashboard built with Gemini, LangChain, ChromaDB, and Streamlit.")

intro_col_1, intro_col_2, intro_col_3 = st.columns(3)

with intro_col_1:
    st.info("Upload TXT/PDF documents")

with intro_col_2:
    st.info("Build or manage the vector database")

with intro_col_3:
    st.info("Ask grounded questions with sources")

st.markdown(
    """
    This app helps you upload documents, index them into ChromaDB, ask questions,
    inspect retrieved sources, and review your session query history.
    """
)

with st.sidebar:
    st.header("Project Overview")

    st.write("Vector DB: ChromaDB")
    st.write("Chat Model: Gemini 2.5 Flash")
    st.write("Embedding Model: Gemini Embedding")

    st.divider()

    st.header("Document Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Files", stats["file_count"])

    with col2:
        st.metric("Chunks", stats["chunk_count"])

    st.metric("Total size", f"{stats['total_size_kb']} KB")

    if stats["chroma_exists"]:
        st.success("Vector database found")
    else:
        st.error("Vector database not found")

    st.caption("Refresh the page after upload, delete, rebuild, or clear actions to update these stats.")

    st.divider()

    st.header("Retrieval Settings")

    top_k = st.slider(
        "Chunks to retrieve",
        min_value=1,
        max_value=8,
        value=5,
        help="Higher values give Gemini more context, but may include less relevant chunks."
    )

    show_debug = st.checkbox(
        "Show retrieval debug panel",
        value=True,
        help="Shows the exact retrieved chunks used before Gemini generated the answer."
    )

    st.caption("Chroma distance score: lower usually means more similar.")

    st.divider()

    st.header("Session History")

    st.metric(
        "Questions asked",
        get_query_history_count(st.session_state)
    )

    if st.button("Clear Query History"):
        clear_query_history(st.session_state)
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

st.divider()

st.subheader("1. Upload Documents")
st.caption("Add TXT or PDF files to the data folder. Uploaded filenames are sanitized automatically.")

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

st.divider()

st.subheader("2. Delete Uploaded Document")
st.caption("Remove a TXT/PDF file from the data folder. Rebuild ChromaDB after deleting a document.")

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

st.divider()

st.subheader("3. Manage Vector Database")
st.caption("Rebuild or clear generated ChromaDB files. Uploaded documents stay inside the data folder.")

manage_col_1, manage_col_2 = st.columns(2)

with manage_col_1:
    st.markdown("**Rebuild ChromaDB**")
    st.write("Use this after adding, deleting, or changing documents.")

    if st.button("Rebuild Vector Database"):
        with st.spinner("Rebuilding vector database. This may take a moment..."):
            result = rebuild_vector_database()

        if result.returncode == 0:
            st.success("Vector database rebuilt successfully.")
            st.code(result.stdout)
            st.info("Refresh the page to update the sidebar dashboard stats.")
        else:
            rebuild_error = create_rebuild_error(result)

            st.error(rebuild_error.user_message)
            st.caption(
                "The app is still running. Wait a minute and try again, "
                "or use a smaller document if the file is very large."
            )

            with st.expander("Rebuild output"):
                if result.stdout:
                    st.code(result.stdout)

                st.write(f"Exit code: {result.returncode}")

with manage_col_2:
    st.markdown("**Clear ChromaDB**")
    st.write("This deletes only generated vector database files.")

    st.warning(
        "Clearing the vector database deletes only generated ChromaDB files. "
        "Your uploaded documents stay inside the data folder."
    )

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

st.divider()

st.subheader("4. Ask Questions")
st.caption("Ask questions grounded in the indexed documents. Answers should come only from retrieved context.")

question = st.text_input(
    "Ask a question",
    placeholder="Type your question here"
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
                session_state=st.session_state,
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

st.divider()

st.subheader("5. Query History")
st.caption("Review questions asked during the current Streamlit session.")

query_history = get_query_history(st.session_state)

if query_history:
    for index, history_item in enumerate(query_history, start=1):
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
