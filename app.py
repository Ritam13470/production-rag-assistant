import os
import subprocess
import sys

import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma

from rag.config import DATA_DIR, DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL, CHAT_MODEL
from rag.embeddings import SafeGoogleEmbeddings
from rag.utils import get_response_text, preview_text, format_source_label

load_dotenv()


PROMPT_TEMPLATE = """
You are a careful and trustworthy RAG assistant.

Answer the user's question using only the context below.

Rules:
1. If the answer is in the context, answer clearly.
2. If the answer is not in the context, say: "I could not find that in the provided documents."
3. Do not invent facts outside the context.
4. Prefer a concise answer first, then add useful detail only if supported by the context.

Context:
{context}

Question:
{question}

Answer:
"""


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


def load_rag_components():
    embeddings = SafeGoogleEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    return vectorstore, llm, prompt


st.set_page_config(
    page_title="Production RAG Assistant",
    page_icon="??",
    layout="wide"
)

st.title("Production RAG Assistant")
st.write("Upload TXT/PDF files, rebuild the vector database, and ask questions using Gemini, LangChain, and ChromaDB.")

with st.sidebar:
    st.header("Project Info")
    st.write("Vector DB: ChromaDB")
    st.write("Chat Model: Gemini 2.5 Flash")
    st.write("Embedding Model: Gemini Embedding")
    st.warning("For learning, upload sample documents only. Avoid private or sensitive files.")

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

    if os.path.exists(DATA_DIR):
        files = [
            file_name
            for file_name in os.listdir(DATA_DIR)
            if file_name.lower().endswith((".txt", ".pdf"))
        ]

        if files:
            for file_name in files:
                st.write(f"- {file_name}")
        else:
            st.write("No TXT/PDF files found.")
    else:
        st.write("Data folder does not exist yet.")

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
        with st.spinner("Searching documents and generating answer..."):
            vectorstore, llm, prompt = load_rag_components()

            scored_docs = vectorstore.similarity_search_with_score(
                question,
                k=top_k
            )

            docs = [
                doc
                for doc, score in scored_docs
            ]

            context = "\n\n".join(
                doc.page_content for doc in docs
            )

            messages = prompt.format_messages(
                context=context,
                question=question
            )

            response = llm.invoke(messages)
            answer = get_response_text(response)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")

        if scored_docs:
            for index, item in enumerate(scored_docs, start=1):
                doc, score = item

                with st.expander(f"Source {index}: {format_source_label(doc, score)}"):
                    st.write(preview_text(doc.page_content))
        else:
            st.write("No sources found.")

        if show_debug:
            st.subheader("Retrieval Debug Panel")

            st.write("This panel shows the chunks retrieved before Gemini generated the answer.")

            for index, item in enumerate(scored_docs, start=1):
                doc, score = item

                st.markdown(f"**Chunk {index}**")
                st.write(format_source_label(doc, score))
                st.code(preview_text(doc.page_content, max_chars=1500))
