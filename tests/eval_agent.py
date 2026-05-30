import uuid
from rag_backend.chunking_indexing import chunk_text, index_document
from rag_backend.agent import run_agent, classify_intent

# ── Load your sample OCR text ─────────────────────────────────────────────────
# Paste some text from your test PDF into tests\sample_text.txt first
try:
    with open("tests/sample_text.txt", "r", encoding="utf-8") as f:
        ocr_text = f.read()
    print(f"Loaded sample_text.txt: {len(ocr_text)} chars\n")
except FileNotFoundError:
    print("WARNING: tests/sample_text.txt not found.")
    print("Using dummy text for evaluation.\n")
    ocr_text = (
        "This paper proposes a transformer-based model for natural language processing. "
        "The authors John Smith and Jane Doe introduce a new attention mechanism. "
        "They evaluate on the GLUE benchmark dataset using BLEU and F1 metrics. "
        "The model outperforms BERT and GPT baselines significantly. "
        "Limitations include high memory usage and lack of multilingual support. "
        "Future work includes scaling to larger datasets and low-resource languages. "
    ) * 40

# ── Index the document ────────────────────────────────────────────────────────
chat_id = str(uuid.uuid4())
chunks = chunk_text(ocr_text)
index_document(chunks, chat_id)
print(f"Indexed {len(chunks)} chunks with chat_id: {chat_id}\n")
print("=" * 70)

# ── Test queries ──────────────────────────────────────────────────────────────
TEST_QUERIES = [
    ("summarise this paper",              "summarise"),
    ("who are the authors",               "extract_entities"),
    ("what dataset was used",             "extract_entities"),
    ("explain the methodology",           "method"),
    ("what are the limitations",          "unsolved"),
    ("explain attention mechanism",       "explain"),
    ("make me 5 bullet point slides",     "slides"),
    ("quiz me on this paper",             "quiz"),
    ("how do I build this",               "build_it"),
    ("suggest similar papers",            "similar_papers"),
    ("compare BERT and GPT",              "compare"),
    ("simplify this for a beginner",      "simplify"),
]

for query, expected_intent in TEST_QUERIES:
    print(f"\nQUERY   : {query}")
    print(f"INTENT  : {classify_intent(query)} (expected: {expected_intent})")
    print("-" * 50)

    try:
        answer = run_agent(query, chat_id, history=[], ocr_text=ocr_text)

        if isinstance(answer, dict) and "slides" in answer:
            slides = answer.get("slides", [])
            print(f"SLIDES  : {len(slides)} slides generated")
            print(f"FORMAT  : {answer.get('format', 'N/A')}")
            print(f"DETAIL  : {answer.get('detail', 'N/A')}")
            for i, s in enumerate(slides[:2], 1):  # preview first 2 slides
                print(f"  Slide {i}: {s.get('title', 'No title')}")
        else:
            preview = str(answer)[:400]
            print(f"RESPONSE: {preview}")
            if len(str(answer)) > 400:
                print(f"          ... ({len(str(answer))} chars total)")

    except Exception as e:
        print(f"ERROR   : {e}")

    print("=" * 70)

print("\nEvaluation complete.")
print("Review each response above and score 1-5 for:")
print("  Relevance | Completeness | Accuracy | Format | Fluency")