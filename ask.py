from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma

from rag.config import DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL, CHAT_MODEL
from rag.embeddings import SafeGoogleEmbeddings
from rag.utils import get_response_text

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


def main():
    print("Starting RAG assistant...")

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

    print("RAG assistant is ready.")
    print("Type your question and press Enter.")
    print("Type exit to quit.")
    print("-" * 60)

    while True:
        question = input("Question: ")

        if question.lower().strip() in ["exit", "quit"]:
            print("Goodbye.")
            break

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

        print()
        print("Answer:")
        print(answer)

        print()
        print("Sources:")
        for index, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "Unknown source")
            page = doc.metadata.get("page")

            if page:
                print(f"{index}. {source}, page {page}")
            else:
                print(f"{index}. {source}")

        print("-" * 60)


if __name__ == "__main__":
    main()
