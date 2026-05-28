import json

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


REFUSAL_TEXT = "I could not find that in the provided documents."


def build_rag_components():
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


def ask_rag(question, vectorstore, llm, prompt, top_k=3):
    docs = vectorstore.similarity_search(question, k=top_k)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    messages = prompt.format_messages(
        context=context,
        question=question
    )

    response = llm.invoke(messages)
    answer = get_response_text(response)

    return answer, docs


def answer_contains_keywords(answer, expected_keywords):
    answer_lower = answer.lower()

    missing_keywords = []

    for keyword in expected_keywords:
        if keyword.lower() not in answer_lower:
            missing_keywords.append(keyword)

    return missing_keywords


def evaluate_case(test_case, vectorstore, llm, prompt):
    question = test_case["question"]
    expected_keywords = test_case["expected_keywords"]
    should_answer = test_case["should_answer"]

    answer, docs = ask_rag(
        question=question,
        vectorstore=vectorstore,
        llm=llm,
        prompt=prompt
    )

    if should_answer:
        missing_keywords = answer_contains_keywords(
            answer=answer,
            expected_keywords=expected_keywords
        )

        passed = len(missing_keywords) == 0

        reason = "All expected keywords found."

        if missing_keywords:
            reason = f"Missing keywords: {missing_keywords}"

    else:
        passed = REFUSAL_TEXT.lower() in answer.lower()
        reason = "Correctly refused to answer from missing context."

        if not passed:
            reason = "Expected refusal, but the assistant answered."

    return {
        "question": question,
        "answer": answer,
        "passed": passed,
        "reason": reason,
        "source_count": len(docs)
    }


def main():
    print("Starting RAG evaluation...")

    with open("eval_questions.json", "r", encoding="utf-8") as file:
        test_cases = json.load(file)

    vectorstore, llm, prompt = build_rag_components()

    passed_count = 0

    for index, test_case in enumerate(test_cases, start=1):
        print("-" * 70)
        print(f"Test {index}: {test_case['question']}")

        result = evaluate_case(
            test_case=test_case,
            vectorstore=vectorstore,
            llm=llm,
            prompt=prompt
        )

        if result["passed"]:
            passed_count += 1
            print("Result: PASS")
        else:
            print("Result: FAIL")

        print(f"Reason: {result['reason']}")
        print(f"Sources retrieved: {result['source_count']}")
        print("Answer:")
        print(result["answer"])

    print("=" * 70)
    print(f"Passed {passed_count} out of {len(test_cases)} tests.")

    if passed_count == len(test_cases):
        print("Evaluation status: SUCCESS")
    else:
        print("Evaluation status: NEEDS IMPROVEMENT")


if __name__ == "__main__":
    main()
