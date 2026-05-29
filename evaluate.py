import json

from rag.pipeline import answer_question

REFUSAL_TEXT = "I could not find that in the provided documents."


def answer_contains_keywords(answer, expected_keywords):
    answer_lower = answer.lower()

    missing_keywords = []

    for keyword in expected_keywords:
        if keyword.lower() not in answer_lower:
            missing_keywords.append(keyword)

    return missing_keywords


def evaluate_case(test_case):
    question = test_case["question"]
    expected_keywords = test_case["expected_keywords"]
    should_answer = test_case["should_answer"]

    try:
        result = answer_question(
            question=question,
            top_k=3
        )

        answer = result.answer
        source_count = len(result.docs)

    except Exception as error:
        return {
            "question": question,
            "answer": "",
            "passed": False,
            "reason": f"Evaluation could not run because the model call failed: {error}",
            "source_count": 0,
            "error": True
        }

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
        "source_count": source_count,
        "error": False
    }


def main():
    print("Starting RAG evaluation...")

    with open("eval_questions.json", "r", encoding="utf-8") as file:
        test_cases = json.load(file)

    passed_count = 0
    error_count = 0

    for index, test_case in enumerate(test_cases, start=1):
        print("-" * 70)
        print(f"Test {index}: {test_case['question']}")

        result = evaluate_case(
            test_case=test_case
        )

        if result["error"]:
            error_count += 1
            print("Result: ERROR")
        elif result["passed"]:
            passed_count += 1
            print("Result: PASS")
        else:
            print("Result: FAIL")

        print(f"Reason: {result['reason']}")
        print(f"Sources retrieved: {result['source_count']}")

        if result["answer"]:
            print("Answer:")
            print(result["answer"])

    print("=" * 70)
    print(f"Passed {passed_count} out of {len(test_cases)} tests.")
    print(f"Errors: {error_count}")

    if error_count > 0:
        print("Evaluation status: BLOCKED BY API ERROR OR QUOTA LIMIT")
    elif passed_count == len(test_cases):
        print("Evaluation status: SUCCESS")
    else:
        print("Evaluation status: NEEDS IMPROVEMENT")


if __name__ == "__main__":
    main()
