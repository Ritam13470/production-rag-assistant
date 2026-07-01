from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma

from rag.config import DB_DIR, COLLECTION_NAME
from rag.database_utils import reset_chroma_client_cache
from rag.providers import build_chat_model, build_embedding_model
from rag.errors import create_rag_error
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
    reset_chroma_client_cache()

    embeddings = build_embedding_model()

    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    return vectorstore


def build_llm():
    llm = build_chat_model()

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
    try:
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

    except Exception as error:
        raise create_rag_error(error) from error
