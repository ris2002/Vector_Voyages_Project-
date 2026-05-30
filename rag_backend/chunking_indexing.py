import os
import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Absolute path – works regardless of cwd ────────────────────────────────────
# This file: <project>/rag_backend/chunking_indexing.py
# DB stored:  <project>/chroma_db/
DB_PATH = str(Path(__file__).resolve().parent.parent / "chroma_db")
os.makedirs(DB_PATH, exist_ok=True)
logger.info(f"ChromaDB path: {DB_PATH}")

# ── Module-level singletons (initialised once per process) ────────────────────
_model         = None
_chroma_client = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _get_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=DB_PATH)
    return _chroma_client


# ── Public API ─────────────────────────────────────────────────────────────────

def chunk_text(
    ocr_text: str,
    chunk_size: int = 400,      # characters per chunk
    overlap: int = 80           # characters of overlap between chunks
) -> list[str]:
    """
    Sliding-window character chunker.
    Splits on sentence boundaries where possible, then slides with overlap.
    Produces far more chunks than a naive paragraph split → better retrieval.
    """
    # First split into sentences (rough but good enough for research papers)
    import re
    sentences = re.split(r'(?<=[.?!])\s+', ocr_text.replace("\n", " "))
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    chunks   = []
    current  = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            # Start next chunk with overlap from end of previous
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = (overlap_text + " " + sentence).strip()

    if current:
        chunks.append(current)

    # Drop anything too short to be meaningful
    chunks = [c for c in chunks if len(c) > 30]
    logger.info(f"chunk_text → {len(chunks)} chunks (size≈{chunk_size}, overlap={overlap})")
    return chunks


def index_document(chunks: list[str], chat_id: str) -> None:
    """
    Embed chunks and persist to ChromaDB.
    Deletes any existing collection first → no stale data ever.
    """
    if not chunks:
        logger.warning("index_document: empty chunk list – nothing indexed")
        return

    client = _get_client()
    model  = _get_model()

    # Always start fresh for this chat session
    try:
        client.delete_collection(name=chat_id)
        logger.info(f"Deleted stale collection '{chat_id}'")
    except Exception:
        pass  # collection didn't exist – fine

    collection = client.create_collection(
        name=chat_id,
        metadata={"hnsw:space": "cosine"}   # cosine similarity for semantic search
    )

    embeddings = model.encode(chunks, show_progress_bar=False).tolist()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[str(i) for i in range(len(chunks))]
    )

    stored = collection.count()
    logger.info(f"✅ Indexed {stored}/{len(chunks)} chunks into '{chat_id}'")

    if stored != len(chunks):
        raise RuntimeError(
            f"Indexing incomplete: stored {stored} of {len(chunks)} chunks"
        )


def retrieve(query: str, chat_id: str, top_k: int = 5) -> list[str]:
    """
    Embed query and return top-k matching chunks.
    Returns [] safely if collection missing or empty.
    """
    client = _get_client()
    model  = _get_model()

    try:
        collection = client.get_collection(name=chat_id)
    except Exception as e:
        logger.error(f"retrieve: collection '{chat_id}' not found – {e}")
        return []

    count = collection.count()
    if count == 0:
        logger.warning(f"retrieve: collection '{chat_id}' is empty")
        return []

    query_embedding = model.encode([query], show_progress_bar=False).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, count)
    )

    chunks = results["documents"][0]
    logger.info(f"retrieve → {len(chunks)} chunks for: '{query[:60]}'")
    return chunks