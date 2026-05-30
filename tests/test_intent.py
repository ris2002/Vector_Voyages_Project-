from rag_backend.agent import classify_intent

cases = [
    ("summarise this paper", "summarise"),
    ("what is the paper about", "summarise"),
    ("who are the authors", "extract_entities"),
    ("what dataset was used", "extract_entities"),
    ("compare BERT and GPT", "compare"),
    ("explain attention mechanism", "explain"),
    ("list the sections", "list_sections"),
    ("explain it like I'm 5", "simplify"),
    ("what does BLEU mean", "jargon"),
    ("what are the limitations", "unsolved"),
    ("how does the model work", "method"),
    ("quiz me on this paper", "quiz"),
    ("how do I build this", "build_it"),
    ("suggest similar papers", "similar_papers"),
    ("make me 5 slides", "slides"),
    ("what is the learning rate", "qa"),
]

passed = 0
failed = 0
for query, expected in cases:
    result = classify_intent(query)
    status = "PASS" if result == expected else "FAIL"
    if result == expected:
        passed += 1
    else:
        failed += 1
    print(f"{status}: '{query}' -- expected '{expected}', got '{result}'")

print(f"\nResults: {passed} passed, {failed} failed out of {len(cases)} tests")