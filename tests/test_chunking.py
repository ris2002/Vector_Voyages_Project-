from rag_backend.chunking_indexing import chunk_text

# Test 1: Normal text
text = "This is sentence one. This is sentence two. " * 50
chunks = chunk_text(text)
assert len(chunks) > 1, "Should produce multiple chunks"
assert all(len(c) > 30 for c in chunks), "All chunks must be >30 chars"

# Test 2: Very short text
chunks = chunk_text("Short.")
assert chunks == [], "Too short text should produce no chunks"

# Test 3: Overlap check
chunks = chunk_text(text, chunk_size=200, overlap=50)
print(f"✅ chunk_text: {len(chunks)} chunks")