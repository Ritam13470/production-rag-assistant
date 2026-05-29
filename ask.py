from rag.errors import RagPipelineError
from rag.pipeline import answer_question


def main():
    print("Starting RAG assistant...")
    print("RAG assistant is ready.")
    print("Type your question and press Enter.")
    print("Type exit to quit.")
    print("-" * 60)

    while True:
        question = input("Question: ")

        if question.lower().strip() in ["exit", "quit"]:
            print("Goodbye.")
            break

        try:
            result = answer_question(
                question=question,
                top_k=3
            )

            print()
            print("Answer:")
            print(result.answer)

            print()
            print("Sources:")
            for index, doc in enumerate(result.docs, start=1):
                source = doc.metadata.get("source", "Unknown source")
                page = doc.metadata.get("page")

                if page:
                    print(f"{index}. {source}, page {page}")
                else:
                    print(f"{index}. {source}")

        except RagPipelineError as error:
            print()
            print("RAG Error:")
            print(error.user_message)

        print("-" * 60)


if __name__ == "__main__":
    main()
