# Vector Voyagers — Research Paper RAG Agentic Chatbot

A multi-agent RAG system for interacting with research papers. Upload a paper, ask questions, get answers grounded in the document.

---

## Project Structure

This is a team project. Responsibilities are split as follows:

| Component | Owner |
|---|---|
| RAG Engineer | Rishil |
| Agent Engineer | TBD |
| Evaluation And Testing | TBD |
| Technical Lead | TBD |

**RAG Engineer (Rishil):**
- Multi-chat Streamlit frontend with isolated session state per chat
- PDF upload pipeline with spinner and status tracking
- OCR pipeline — pdfplumber for digital PDFs, Gemini Flash vision fallback for figure-heavy pages
- Title extraction — PDF metadata → Ollama DeepSeek → filename fallback
- File hash caching — same PDF never processed twice
- Chunking, embedding, and ChromaDB vector storage
- Session state architecture design

**Agent Engineer:** TBD

---

## Features (Current)

- Multi-chat sidebar — each chat is isolated with its own PDF and history
- PDF upload with spinner — processes on upload, not on first query
- OCR pipeline — pdfplumber for digital PDFs, Gemini Flash vision fallback for figure-heavy pages
- Title extraction — PDF metadata → Ollama DeepSeek fallback → filename fallback
- Session state architecture — chat history persists across reruns
- File hash caching — same PDF never processed twice

---

## Tech Stack

| Library | Purpose |
|---|---|
| Streamlit | Frontend UI |
| pdfplumber | Text extraction from digital PDFs |
| pdf2image | Convert scanned PDF pages to images |
| Gemini Flash | Vision fallback for figure-heavy pages |
| Ollama (DeepSeek-r1:7b) | Title extraction, chat LLM |
| ChromaDB | Vector storage (one collection per chat) |
| sentence-transformers | Chunk embeddings |
| PyMuPDF (fitz) | PDF metadata extraction |
| hashlib | File hash caching |

---

## Setup

### Prerequisites

- Python 3.11+
- Ollama installed — [ollama.com](https://ollama.com)
- Poppler installed (for pdf2image)

```bash
# Mac
brew install poppler

# Linux
sudo apt install poppler-utils
```

### Install dependencies

```bash
pip install streamlit pdfplumber pdf2image pymupdf chromadb \
            sentence-transformers google-generativeai ollama \
            python-dotenv
```

### Pull Ollama model

```bash
ollama pull deepseek-r1:7b
```

### Environment variables

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

Gemini is used only for vision (figure-heavy pages). Get a free key at [aistudio.google.com](https://aistudio.google.com).

### Run

```bash
streamlit run app.py
```

---

## File Structure

```
vector_voyagers/
├── app.py              # Streamlit UI
├── ocr.py              # OCR pipeline (pdfplumber + Gemini vision fallback)
├── utils.py            # Title extraction
├── rag.py              # Chunking, embedding, ChromaDB indexing
├── .env                # API keys (never commit this)
└── README.md
```

---

## Session State Schema

```python
{
    "chats": {
        "uuid": {
            "title": "Attention Is All You Need",
            "pdf_name": "transformer.pdf",
            "pdf_hash": "a3f9bc...",
            "ocr_text": "...",
            "messages": [],
            "status": "pending | processing | ready"
        }
    },
    "active_chat_id": "uuid"
}
```

---

## Notes for Agent Engineer

- Each chat has its own isolated ChromaDB collection keyed by `chat_id` (UUID)
- OCR text is stored in `session_state.chats[chat_id]["ocr_text"]` — accessible directly
- Status flow: `pending → processing → ready`
- LLM is swappable — currently Ollama DeepSeek locally, change model in `utils.py` and `rag.py`
- Gemini is only used for vision — swap in any vision LLM by updating `ocr.py`
- `@st.cache_data` on `run_ocr` — same PDF never re-processed across reruns
