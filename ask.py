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
            print(f"{index}. {source}")

        print("-" * 60)


if __name__ == "__main__":
    main()
