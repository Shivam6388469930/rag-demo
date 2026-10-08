# evaluate.py
from rag_graph import build_graph

TEST_CASES = [
    {"question": "What is RAG?", "expected_topic": "RAG", "should_answer": True},
    {
        "question": "How does Ollama help developers?",
        "expected_topic": "Ollama",
        "should_answer": True,
    },
    {
        "question": "What is the capital of France?",
        "expected_topic": None,
        "should_answer": False,
    },
    {
        "question": "What vector databases exist?",
        "expected_topic": "Vector databases",
        "should_answer": True,
    },
]


def evaluate(app):
    passed = 0
    failed = 0

    for case in TEST_CASES:
        result = app.invoke(
            {
                "question": case["question"],
                "documents": [],
                "generation": "",
                "retries": 0,
            }
        )

        answered = "i don't know" not in result["generation"].lower()
        correct = answered == case["should_answer"]

        status = "PASS" if correct else "FAIL"
        if correct:
            passed += 1
        else:
            failed += 1

        print(f"[{status}] {case['question']}")

    print(f"\nResult: {passed}/{passed+failed} passed")
    return passed / (passed + failed)


if __name__ == "__main__":
    app = build_graph()
    score = evaluate(app)
