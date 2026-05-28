import streamlit as st
from dotenv import load_dotenv

from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma

load_dotenv()

DB_DIR = "chroma_db"
COLLECTION_NAME = "rag_documents"
EMBEDDING_MODEL = "gemini-embedding-2-preview"
CHAT_MODEL = "gemini-2.5-flash"


class SafeGoogleEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model=model_name
        )

    def embed_documents(self, texts):
        vectors = []

        for text in texts:
            vector = self.embedding_model.embed_query(text)
            vectors.append(vector)

        return vectors

    def embed_query(self, text):
        return self.embedding_model.embed_query(text)


PROMPT_TEMPLATE = """
You are a helpful RAG assistant.

Answer the user's question using only the context below.

Rules:
1. If the answer is in the context, answer clearly.
2. If the answer is not in the context, say: "I could not find that in the provided documents."
3. Do not invent facts outside the context.

Context:
{context}

Question:
{question}

Answer:
"""


def get_response_text(response):
    content = response.content

    if isinstance(content, list):
        text = ""

        for part in content:
            if isinstance(part, dict):
                text += part.get("text", "")
            else:
                text += str(part)

        return text

    return str(content)


@st.cache_resource
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
st.write("Ask questions from your TXT and PDF documents using Gemini, LangChain, and ChromaDB.")

with st.sidebar:
    st.header("Project Info")
    st.write("Vector DB: ChromaDB")
    st.write("Chat Model: Gemini 2.5 Flash")
    st.write("Embedding Model: Gemini Embedding")
    st.info("Run python ingest.py after adding or changing files in the data folder.")

question = st.text_input(
    "Ask a question",
    placeholder="Example: What backend technologies does CharacterForge AI use?"
)

if st.button("Ask"):
    if not question.strip():
        st.warning("Please type a question first.")
    else:
        with st.spinner("Searching documents and generating answer..."):
            vectorstore, llm, prompt = load_rag_components()

            docs = vectorstore.similarity_search(question, k=3)

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

        if docs:
            for index, doc in enumerate(docs, start=1):
                source = doc.metadata.get("source", "Unknown source")
                page = doc.metadata.get("page")

                if page:
                    st.write(f"{index}. {source} - page {page}")
                else:
                    st.write(f"{index}. {source}")
        else:
            st.write("No sources found.")
