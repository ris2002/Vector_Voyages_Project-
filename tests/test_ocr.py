import os
from rag_backend.ocr import run_ocr

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

# ── Load sample PDF ───────────────────────────────────────────────────────────
PDF_PATH = "tests/sample.pdf"

if not os.path.exists(PDF_PATH):
    print("ERROR: tests/sample.pdf not found.")
    print("Please copy any small research PDF into the tests folder:")
    print('  copy "C:\\Users\\YourName\\Downloads\\paper.pdf" tests\\sample.pdf')
    exit(1)

print(f"Loading PDF: {PDF_PATH}")
with open(PDF_PATH, "rb") as f:
    pdf_bytes = f.read()

print(f"PDF size: {len(pdf_bytes)} bytes\n")

# ── Test 1: OCR runs without crashing ────────────────────────────────────────
try:
    text = run_ocr(pdf_bytes)
    check("OCR runs without crashing", True)
except Exception as e:
    check(f"OCR runs without crashing (error: {e})", False)
    print("Cannot continue tests — OCR failed.")
    exit(1)

# ── Test 2: Returns a string ──────────────────────────────────────────────────
check("OCR returns a string", isinstance(text, str))

# ── Test 3: Returns meaningful content ───────────────────────────────────────
check("OCR returns more than 100 chars", len(text) > 100)

# ── Test 4: No empty result ───────────────────────────────────────────────────
check("OCR result is not empty", len(text.strip()) > 0)

# ── Test 5: Contains real words ──────────────────────────────────────────────
common_words = ["the", "a", "of", "and", "in", "is", "to"]
found = any(word in text.lower() for word in common_words)
check("OCR result contains real words", found)

# ── Save extracted text to sample_text.txt for use in eval_agent.py ──────────
output_path = "tests/sample_text.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(text)
print(f"\nSaved extracted text to {output_path} ({len(text)} chars)")
print(f"Preview (first 300 chars):\n{text[:300]}")

print(f"\nResults: {passed} passed, {failed} failed out of {passed + failed} tests")