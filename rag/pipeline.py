from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma

from rag.config import DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL, CHAT_MODEL
from rag.embeddings import SafeGoogleEmbeddings
from rag.prompts import RAG_PROMPT_TEMPLATE
from rag.utils import get_response_text

load_dotenv()


@dataclass
class RagResult:
    question: str
    answer: str
    docs: list[Document]
    scored_docs: list[tuple[Document, float]]
    context: str


def build_vectorstore():
    embeddings = SafeGoogleEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    return vectorstore


def build_llm():
    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        temperature=0
    )

    return llm


def build_prompt():
    prompt = ChatPromptTemplate.from_template(
        RAG_PROMPT_TEMPLATE
    )

    return prompt


def build_rag_components():
    vectorstore = build_vectorstore()
    llm = build_llm()
    prompt = build_prompt()

    return vectorstore, llm, prompt


def answer_question(question, top_k=3):
    vectorstore, llm, prompt = build_rag_components()

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

    return RagResult(
        question=question,
        answer=answer,
        docs=docs,
        scored_docs=scored_docs,
        context=context
    )
