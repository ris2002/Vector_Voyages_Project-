import uuid
from rag_backend.chunking_indexing import chunk_text, index_document, retrieve
from rag_backend.agent import run_agent

passed = 0
failed = 0

def check(label, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"PASS: {label}")
    else:
        failed += 1
        print(f"FAIL: {label}")

# ── Test 1: chunk_text with very short text ──────────────────────────────────
chunks = chunk_text("Short.")
check("Short text produces no chunks", chunks == [])

# ── Test 2: chunk_text with normal text ──────────────────────────────────────
long_text = "This is a test sentence about machine learning. " * 60
chunks = chunk_text(long_text)
check("Normal text produces multiple chunks", len(chunks) > 1)
check("All chunks are longer than 30 chars", all(len(c) > 30 for c in chunks))

# ── Test 3: index_document with empty list ───────────────────────────────────
try:
    index_document([], "test_empty_session")
    check("Empty chunk list does not crash", True)
except Exception as e:
    check(f"Empty chunk list does not crash (error: {e})", False)

# ── Test 4: retrieve from non-existent collection ────────────────────────────
try:
    results = retrieve("anything", "nonexistent_chat_id_xyz_999")
    check("Retrieve from missing collection returns []", results == [])
except Exception as e:
    check(f"Retrieve from missing collection does not crash (error: {e})", False)

# ── Test 5: run_agent with no ocr_text ───────────────────────────────────────
result = run_agent("summarise this", "fake_id_abc", [], ocr_text="")
check("run_agent with no OCR returns helpful message",
      "ocr" in result.lower() or "upload" in result.lower() or "available" in result.lower())

# ── Test 6: retrieve with gibberish query ────────────────────────────────────
chat_id = str(uuid.uuid4())
sample = "The transformer architecture uses attention mechanisms. " * 30
chunks = chunk_text(sample)
index_document(chunks, chat_id)
results = retrieve("zzzzz gibberish xyz123", chat_id, top_k=3)
check("Gibberish query returns results without crashing", isinstance(results, list))

# ── Test 7: chunk overlap works ──────────────────────────────────────────────
chunks = chunk_text(long_text, chunk_size=200, overlap=50)
check("Chunking with overlap produces chunks", len(chunks) > 0)

print(f"\nResults: {passed} passed, {failed} failed out of {passed + failed} tests")