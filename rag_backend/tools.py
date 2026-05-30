import os
import re
import logging
from groq import Groq
from dotenv import load_dotenv
from .chunking_indexing import retrieve

load_dotenv()
logger = logging.getLogger(__name__)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Model chain ───────────────────────────────────────────────────────────────
# Seed list — only confirmed-live models as of May 2025.
# At runtime _build_model_chain() extends this with any additional active
# chat models returned by the Groq /v1/models endpoint, so the chain stays
# fresh even if Groq adds or removes models after this code was written.
_SEED_MODELS = [
    "llama-3.1-8b-instant",          # fastest, production
    "llama-3.3-70b-versatile",       # best quality, production
    "meta-llama/llama-4-scout-17b-16e-instruct",   # preview, fast
    "meta-llama/llama-4-maverick-17b-128e-instruct", # preview, quality
    "qwen/qwen-3-32b",               # preview fallback
]

# Models known to be decommissioned — never retry these
_DEAD_MODELS = {
    "llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768",
    "llama-3.1-70b-versatile", "llama-3.1-70b-specdec",
    "gemma2-9b-it", "gemma-7b-it",
    "llama3-groq-8b-8192-tool-use-preview",
    "llama3-groq-70b-8192-tool-use-preview",
}


def _build_model_chain() -> list[str]:
    """
    Query Groq /v1/models for active chat models and merge with seed list.
    Falls back to seed list silently if the API call fails.
    """
    import requests as _req
    try:
        resp = _req.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY', '')}"},
            timeout=3,
        )
        if resp.status_code == 200:
            live_ids = {
                m["id"] for m in resp.json().get("data", [])
                if m.get("active") and m.get("object") == "model"
                # exclude audio / vision-only / guard models
                and not any(skip in m["id"] for skip in
                    ["whisper", "guard", "tts", "vision", "embed"])
            }
            # seed order first, then any extra live models not in seed
            chain = [m for m in _SEED_MODELS if m in live_ids and m not in _DEAD_MODELS]
            extras = sorted(live_ids - set(_SEED_MODELS) - _DEAD_MODELS)
            return chain + extras
    except Exception:
        pass
    return [m for m in _SEED_MODELS if m not in _DEAD_MODELS]


GROQ_MODEL_CHAIN: list[str] = _build_model_chain()
logger.info(f"GROQ_MODEL_CHAIN: {GROQ_MODEL_CHAIN}")


# ── Core LLM call with automatic model fallback ───────────────────────────────

def _llm(prompt: str, system: str = "", max_tokens: int = 700) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_error = None
    for model in GROQ_MODEL_CHAIN:
        try:
            logger.info(f"_llm: trying model '{model}'")
            response = groq_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                timeout=30,
            )
            result = response.choices[0].message.content.strip()
            logger.info(f"_llm: success with model '{model}'")
            return result

        except Exception as e:
            err_str = str(e).lower()
            if any(kw in err_str for kw in ["rate_limit", "rate limit", "429", "quota", "tokens per day", "tpd", "rpm", "tpm", "decommissioned", "deprecated", "model_not_found", "model not found", "not supported", "invalid_model"]):
                logger.warning(f"_llm: model '{model}' rate-limited, trying next. Error: {e}")
                last_error = e
                continue
            logger.error(f"_llm: model '{model}' failed with non-rate-limit error: {e}")
            raise RuntimeError(f"LLM call failed: {e}")

    logger.error("_llm: all models in fallback chain exhausted")
    raise RuntimeError(
        "All Groq models are currently rate-limited. "
        "Please wait a few minutes and try again, or upgrade your Groq plan at "
        "https://console.groq.com/settings/billing\n\n"
        f"Last error: {last_error}"
    )


# ── Keyword scan helper ────────────────────────────────────────────────────────

def _keyword_scan(query: str, ocr_text: str, max_hits: int = 4) -> list[str]:
    """
    Returns paragraphs from full OCR text that contain keywords from the query.
    Supplements vector search to ensure nothing is missed.
    """
    if not ocr_text:
        return []

    stopwords = {
        "what", "which", "where", "when", "does", "that", "this",
        "with", "from", "have", "about", "into", "their", "there",
        "were", "been", "they", "than", "then", "some", "will",
        "would", "could", "should", "your", "more", "also", "how",
        "did", "paper", "study", "research", "article",
    }
    keywords = [
        w.lower() for w in re.split(r"\W+", query)
        if len(w) > 3 and w.lower() not in stopwords
    ]
    if not keywords:
        return []

    paragraphs = [p.strip() for p in ocr_text.split("\n\n") if len(p.strip()) > 80]
    hits = []
    for para in paragraphs:
        if any(kw in para.lower() for kw in keywords):
            hits.append(para)
        if len(hits) >= max_hits:
            break
    return hits


# ── SUMMARISE ─────────────────────────────────────────────────────────────────

def summarise_paper(ocr_text: str) -> str:
    """
    Summarise the full paper in a single Groq call (up to 24000 chars).
    Single call avoids rate limit issues from chained requests.
    """
    text = ocr_text.strip()
    if not text:
        return "No text was extracted from this PDF. Please try re-uploading."

    snippet = text[:6000]
    system = (
        "You are a research assistant. Read the paper text and write a thorough summary.\n"
        "Use this exact structure:\n\n"
        "**Overview:** What is this paper about?\n\n"
        "**Key Findings / Contributions:** What are the main results or contributions?\n\n"
        "**Methods:** How was the research conducted?\n\n"
        "**Conclusions:** What conclusions or implications were drawn?\n\n"
        "Be specific — include actual model names, datasets, metrics, and numbers from the paper. "
        "Do not say the document lacks information. Do not be vague or generic."
    )
    return _llm(f"Paper text:\n{snippet}", system=system)


# ── GENERAL QA WITH CITATIONS ─────────────────────────────────────────────────

def answer_from_document(
    query: str,
    chat_id: str,
    history: list[dict],
    ocr_text: str = ""
) -> str:
    chunks = retrieve(query, chat_id, top_k=8)
    doc_start = ocr_text[:2000] if ocr_text else ""
    keyword_hits = _keyword_scan(query, ocr_text, max_hits=4)

    if not chunks and not doc_start and not keyword_hits:
        return "Could not retrieve any context from the document. Try re-uploading the PDF."

    context_parts = []
    if doc_start:
        context_parts.append(f"[Document Start]\n{doc_start}")
    if chunks:
        context_parts.append("[Relevant Sections]\n" + "\n\n---\n\n".join(chunks))
    if keyword_hits:
        context_parts.append("[Keyword Matched Sections]\n" + "\n\n---\n\n".join(keyword_hits))

    context = "\n\n".join(context_parts)

    history_text = ""
    for msg in history[-6:]:
        if msg["role"] in ("user", "assistant"):
            history_text += f"{msg['role'].capitalize()}: {msg['content']}\n"

    system = (
        "You are a research assistant helping a user understand a research paper.\n\n"
        "INSTRUCTIONS:\n"
        "1. Use the conversation history to understand follow-up questions.\n"
        "2. Answer using ONLY the paper content provided below — not general knowledge.\n"
        "3. Be specific — include actual names, numbers, methods, and results from the paper.\n"
        "4. After your answer, on a new line write:\n"
        "   📄 *Sources: [first 15 words of chunk 1] | [first 15 words of chunk 2]*\n"
        "5. Do NOT say the document lacks information if any relevant content exists.\n"
        "6. If the question is a follow-up and context is thin, say what you found "
        "and suggest a more specific rephrasing."
    )
    prompt = (
        f"--- PAPER CONTENT ---\n{context}\n--- END PAPER CONTENT ---\n\n"
        f"{history_text}"
        f"User question: {query}"
    )
    return _llm(prompt, system=system)


# ── MULTI-STEP REASONING ──────────────────────────────────────────────────────

def answer_complex_question(
    query: str,
    chat_id: str,
    history: list[dict],
    ocr_text: str = ""
) -> str:
    decompose_system = (
        "You are a research assistant. Break the user's question into 2-3 simpler "
        "sub-questions that together would fully answer it. "
        "Return ONLY a numbered list. No explanations, no preamble."
    )
    sub_questions_raw = _llm(f"Question: {query}", system=decompose_system)
    logger.info(f"answer_complex_question: sub-questions:\n{sub_questions_raw}")

    sub_contexts = []
    lines = [l.strip() for l in sub_questions_raw.split("\n") if l.strip()]
    for line in lines[:3]:
        clean = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        if not clean:
            continue
        sub_chunks = retrieve(clean, chat_id, top_k=4)
        sub_kw = _keyword_scan(clean, ocr_text, max_hits=2)
        sub_context = "\n\n".join(sub_chunks + sub_kw)
        if sub_context:
            sub_contexts.append(f"Sub-question: {clean}\nContext:\n{sub_context}")

    if not sub_contexts:
        logger.warning("answer_complex_question: no sub-contexts found, falling back to QA")
        return answer_from_document(query, chat_id, history, ocr_text=ocr_text)

    doc_start = ocr_text[:3000] if ocr_text else ""
    combined = "\n\n---\n\n".join(sub_contexts)

    system = (
        "You are a research assistant. Using the retrieved context for each sub-question, "
        "write one comprehensive answer to the original question.\n\n"
        "INSTRUCTIONS:\n"
        "1. Answer using ONLY the paper content provided — not general knowledge.\n"
        "2. Be specific — include actual names, numbers, methods, and results.\n"
        "3. Synthesise across all sub-questions into one flowing answer.\n"
        "4. After your answer, on a new line write:\n"
        "   📄 *Sources: [first 15 words of key chunk 1] | [first 15 words of key chunk 2]*\n"
        "5. Do not be vague or generic."
    )
    prompt = (
        f"Original question: {query}\n\n"
        f"Document start:\n{doc_start}\n\n"
        f"Retrieved context by sub-question:\n{combined}\n\n"
        "Write a single comprehensive answer to the original question."
    )
    return _llm(prompt, system=system)


# ── EXTRACT ENTITIES ──────────────────────────────────────────────────────────

def extract_entities(ocr_text: str) -> str:
    snippet = ocr_text[:5000]
    system = (
        "You are a research assistant. Extract key entities from the paper text.\n"
        "Return them in this exact format:\n\n"
        "**Authors:** ...\n\n"
        "**Datasets:** ...\n\n"
        "**Models / Methods:** ...\n\n"
        "**Evaluation Metrics:** ...\n\n"
        "**Baselines:** ...\n\n"
        "If something is not mentioned, write 'Not mentioned'."
    )
    return _llm(f"Paper text:\n{snippet}", system=system)


# ── COMPARE ───────────────────────────────────────────────────────────────────

def compare_concepts(
    term_a: str,
    term_b: str,
    chat_id: str,
    ocr_text: str = ""
) -> str:
    chunks_a = retrieve(term_a, chat_id, top_k=4)
    chunks_b = retrieve(term_b, chat_id, top_k=4)

    if len(chunks_a) < 2:
        chunks_a += _keyword_scan(term_a, ocr_text, max_hits=3)
    if len(chunks_b) < 2:
        chunks_b += _keyword_scan(term_b, ocr_text, max_hits=3)

    context_a = "\n\n".join(chunks_a) if chunks_a else "No relevant content found."
    context_b = "\n\n".join(chunks_b) if chunks_b else "No relevant content found."

    system = (
        "You are a research assistant. Compare the two concepts using ONLY the paper context.\n"
        "Use this exact structure:\n\n"
        f"**{term_a}:** (what the paper says)\n\n"
        f"**{term_b}:** (what the paper says)\n\n"
        "**Key Differences:** (based on the paper)\n\n"
        "📄 *Sources: [first 15 words of key source chunk]*\n\n"
        "Be specific. Do not use general knowledge."
    )
    prompt = (
        f"Context about '{term_a}':\n{context_a}\n\n"
        f"Context about '{term_b}':\n{context_b}\n\n"
        f"Compare '{term_a}' and '{term_b}' as described in the paper."
    )
    return _llm(prompt, system=system)


# ── EXPLAIN ───────────────────────────────────────────────────────────────────

def explain_concept(term: str, chat_id: str, ocr_text: str = "") -> str:
    chunks = retrieve(term, chat_id, top_k=5)
    if len(chunks) < 2:
        chunks += _keyword_scan(term, ocr_text, max_hits=4)

    if not chunks:
        return (
            f"The term '{term}' does not appear to be discussed in this paper. "
            "Try rephrasing or asking about a different concept."
        )

    context = "\n\n".join(chunks)
    system = (
        "You are a research assistant. Explain the concept using ONLY the paper context below.\n"
        "Keep the explanation clear, specific, and grounded in what the paper says.\n"
        "After the explanation add:\n"
        "📄 *Sources: [first 15 words of the chunk used]*\n\n"
        "Do not use general knowledge.\n\n"
        f"--- PAPER CONTEXT ---\n{context}\n--- END CONTEXT ---"
    )
    return _llm(f"Explain: {term}", system=system)


# ── LIST SECTIONS ─────────────────────────────────────────────────────────────

def list_sections(ocr_text: str) -> str:
    snippet = ocr_text[:5000]
    system = (
        "You are a research assistant. Identify the main sections or topics of this paper.\n"
        "If the paper has formal section headings, list those.\n"
        "If not, describe the main topics covered in order.\n"
        "Return a clean numbered list with a one-sentence description of each."
    )
    return _llm(f"Paper text:\n{snippet}", system=system)


# ══════════════════════════════════════════════════════════════════════════════
# NEW TOOLS
# ══════════════════════════════════════════════════════════════════════════════

# ── SIMPLIFY (ELI5) ───────────────────────────────────────────────────────────

def simplify_paper(ocr_text: str) -> str:
    """
    Explain the paper in very simple language for a complete beginner.
    """
    snippet = ocr_text[:6000]
    system = (
        "You are a friendly teacher explaining a research paper to a curious 15-year-old "
        "with no technical background.\n\n"
        "INSTRUCTIONS:\n"
        "- Use everyday language. Avoid jargon — if you must use a technical term, define it immediately.\n"
        "- Use simple analogies and real-world comparisons.\n"
        "- Structure your response as:\n\n"
        "**What problem were they trying to solve?**\n\n"
        "**What did they do to solve it?** (in simple terms)\n\n"
        "**What did they find out?**\n\n"
        "**Why does this matter in real life?**\n\n"
        "Keep it engaging, clear, and under 400 words."
    )
    return _llm(f"Paper text:\n{snippet}", system=system)


# ── DEFINE JARGON ─────────────────────────────────────────────────────────────

def define_jargon(term: str, chat_id: str, ocr_text: str = "") -> str:
    """
    If user asks for all difficult words — scan paper and explain them all.
    If user asks about a specific term — explain just that one.
    """
    import re as _re

    # Detect if this is a bulk "explain all hard words" request
    bulk_triggers = [
        r"\bdifficult word",
        r"\bhard word",
        r"\btechnical word",
        r"\bcomplex word",
        r"\bjargon\b",
        r"\bterminolog",
        r"\bvocabular",
        r"\bglossary\b",
        r"\ball.*word",
        r"\bword.*simple",
        r"\bsimple.*word",
        r"\bword.*paper",
    ]
    is_bulk = any(_re.search(p, term.lower()) for p in bulk_triggers) or len(term.split()) > 5

    if is_bulk:
        # ── BULK: find and explain all hard words in the paper ────────────────
        snippet = ocr_text[:12000] if ocr_text else ""
        if not snippet:
            return "No paper text available. Please re-upload the PDF."

        system = (
            "You are a friendly research assistant helping someone understand a paper.\n"
            "The user wants ALL difficult, technical, or domain-specific words explained simply.\n\n"
            "TASK:\n"
            "1. Read the paper text carefully.\n"
            "2. Pick 10-15 of the most difficult or technical words/phrases ACTUALLY used in the paper.\n"
            "3. For EACH word/phrase, write:\n"
            "   **[Word/Phrase]**\n"
            "   Simple meaning: [1 sentence plain English definition]\n"
            "   In this paper: [1 sentence on how it is used in this paper specifically]\n"
            "   Example: [one short real-world analogy or example]\n\n"
            "RULES:\n"
            "- Only use words that ACTUALLY APPEAR in the paper text.\n"
            "- Do NOT explain common English words.\n"
            "- Do NOT give a general explanation — always tie it to this paper.\n"
            "- Keep each explanation under 60 words total.\n"
            "- Order from most common to most technical.\n"
            "- Do NOT add any intro or outro — just the word list."
        )
        return _llm(f"Paper text:\n{snippet}", system=system, max_tokens=2000)

    else:
        # ── SINGLE TERM: explain just that one word ───────────────────────────
        chunks = retrieve(term, chat_id, top_k=5)
        if len(chunks) < 2:
            chunks += _keyword_scan(term, ocr_text, max_hits=4)
        context = "\n\n".join(chunks) if chunks else ocr_text[:3000]

        system = (
            "You are a friendly research assistant.\n"
            "Explain the technical term below in SIMPLE language.\n\n"
            "FORMAT:\n"
            "**[Term]**\n"
            "Simple meaning: [1-2 sentences plain English — no jargon]\n"
            "In this paper: [1-2 sentences on how this paper uses the term]\n"
            "Real-world analogy: [one simple comparison anyone can understand]\n\n"
            "RULES:\n"
            "- No bullet points, no numbered lists — just the four lines above.\n"
            "- Keep total response under 100 words.\n"
            "- Base the \'In this paper\' section strictly on the context below.\n\n"
            f"--- PAPER CONTEXT ---\n{context}\n--- END CONTEXT ---"
        )
        return _llm(f"Explain this term simply: {term}", system=system, max_tokens=300)


# ── UNSOLVED PROBLEMS ─────────────────────────────────────────────────────────

def find_unsolved_problems(ocr_text: str, chat_id: str) -> str:
    """
    Identify limitations and open problems the paper acknowledges or implies.
    """
    # Focus on conclusions, limitations, future work sections
    chunks = retrieve("limitations future work open problems unsolved", chat_id, top_k=6)
    kw_hits = _keyword_scan("limitation future unsolved open problem", ocr_text, max_hits=5)
    doc_end = ocr_text[-5000:] if len(ocr_text) > 5000 else ocr_text

    context_parts = []
    if chunks:
        context_parts.append("[Relevant Sections]\n" + "\n\n---\n\n".join(chunks))
    if kw_hits:
        context_parts.append("[Keyword Hits]\n" + "\n\n---\n\n".join(kw_hits))
    context_parts.append(f"[Paper Ending]\n{doc_end}")
    context = "\n\n".join(context_parts)

    system = (
        "You are a research analyst. Identify what problems remain unsolved after this paper.\n\n"
        "Use this structure:\n\n"
        "**Limitations the Authors Admit:**\n"
        "(direct limitations stated in the paper)\n\n"
        "**Open Problems & Future Work:**\n"
        "(what the authors say needs more research)\n\n"
        "**Implied Gaps:**\n"
        "(gaps you can infer from the methodology or results — briefly)\n\n"
        "Be specific and grounded in the paper. Do not make things up.\n"
        "📄 *Sources: [first 15 words of key chunk]*"
    )
    return _llm(f"Paper context:\n{context}", system=system)


# ── EXPLAIN METHOD / APPROACH ─────────────────────────────────────────────────

def explain_method(ocr_text: str, chat_id: str) -> str:
    """
    Explain the core methodology or approach the paper uses — clearly and in depth.
    """
    chunks = retrieve("method approach methodology architecture model training", chat_id, top_k=8)
    kw_hits = _keyword_scan("method approach propose algorithm model architecture", ocr_text, max_hits=4)
    doc_start = ocr_text[:2500]

    context_parts = [f"[Paper Start]\n{doc_start}"]
    if chunks:
        context_parts.append("[Method Sections]\n" + "\n\n---\n\n".join(chunks))
    if kw_hits:
        context_parts.append("[Keyword Hits]\n" + "\n\n---\n\n".join(kw_hits))
    context = "\n\n".join(context_parts)

    system = (
        "You are a research assistant. Explain the methodology or approach of this paper clearly.\n\n"
        "Use this structure:\n\n"
        "**Core Idea / Approach:**\n"
        "(What is the central technique or method?)\n\n"
        "**How It Works — Step by Step:**\n"
        "(Walk through the process in logical order)\n\n"
        "**Key Design Choices:**\n"
        "(Important decisions the authors made and why)\n\n"
        "**What Makes This Different:**\n"
        "(How does this differ from prior approaches?)\n\n"
        "📄 *Sources: [first 15 words of key chunk]*\n\n"
        "Be specific — use actual names, formulas, datasets, and numbers from the paper."
    )
    return _llm(f"Paper context:\n{context}", system=system)


# ── GENERATE QUIZ ─────────────────────────────────────────────────────────────

def generate_quiz(ocr_text: str, quiz_count: int = 5) -> str:
    """
    Generate exactly quiz_count MCQ questions. Called as generate_quiz(ocr_text, quiz_count=N).
    """
    quiz_count = max(1, min(quiz_count, 20))
    snippet    = ocr_text[:6000]
    if quiz_count <= 3:
        difficulty = f"All {quiz_count} at medium difficulty."
    else:
        easy   = max(1, round(quiz_count * 0.30))
        hard   = max(1, round(quiz_count * 0.20))
        medium = quiz_count - easy - hard
        difficulty = f"Vary difficulty: {easy} easy, {medium} medium, {hard} hard."
    system = (
        f"You are an educational assistant. Create EXACTLY {quiz_count} multiple-choice "
        f"quiz questions based on the research paper below.\n\n"
        "FORMAT (repeat for every question):\n"
        "**Q[N]. [Question text]**\n"
        "A) ...\n"
        "B) ...\n"
        "C) ...\n"
        "D) ...\n"
        "✅ **Answer: [Letter] — [1-sentence explanation]**\n\n"
        "Rules:\n"
        f"- Output EXACTLY {quiz_count} questions — count carefully.\n"
        "- Cover different aspects: concepts, results, methods, datasets, authors.\n"
        "- Wrong options must be plausible, not obviously silly.\n"
        "- Base every question strictly on the paper — no invented facts.\n"
        f"- {difficulty}"
    )
    max_tokens = min(300 + quiz_count * 220, 4000)
    return _llm(f"Paper text:\n{snippet}", system=system, max_tokens=max_tokens)


# ── BUILD IT (IMPLEMENTATION GUIDE) ──────────────────────────────────────────

def build_it_guide(ocr_text: str, chat_id: str) -> str:
    """
    Tells the user how to practically implement or build what this paper proposes.
    """
    chunks = retrieve("implementation code architecture training dataset evaluation", chat_id, top_k=8)
    kw_hits = _keyword_scan("implement train dataset hyperparameter code reproduce", ocr_text, max_hits=4)
    doc_start = ocr_text[:2500]

    context_parts = [f"[Paper Start]\n{doc_start}"]
    if chunks:
        context_parts.append("[Technical Sections]\n" + "\n\n---\n\n".join(chunks))
    if kw_hits:
        context_parts.append("[Keyword Hits]\n" + "\n\n---\n\n".join(kw_hits))
    context = "\n\n".join(context_parts)

    system = (
        "You are a senior ML engineer helping someone reproduce or build what a paper proposes.\n\n"
        "Use this structure:\n\n"
        "**What You're Building:**\n"
        "(1-2 sentence description of the system/model)\n\n"
        "**Prerequisites & Tools:**\n"
        "(libraries, frameworks, hardware — be specific where the paper mentions them)\n\n"
        "**Data:**\n"
        "(datasets used, how to get them, preprocessing steps if mentioned)\n\n"
        "**Step-by-Step Implementation Plan:**\n"
        "(numbered steps from setup to evaluation — grounded in the paper)\n\n"
        "**Key Hyperparameters & Settings:**\n"
        "(exact values from the paper where available)\n\n"
        "**Potential Pitfalls:**\n"
        "(things to watch out for based on paper's limitations section)\n\n"
        "📄 *Sources: [first 15 words of key chunk]*\n\n"
        "Only use information from the paper. Do not invent steps or values."
    )
    return _llm(f"Paper context:\n{context}", system=system, max_tokens=800)


# ── SUGGEST SIMILAR PAPERS ────────────────────────────────────────────────────

def suggest_similar_papers(ocr_text: str, chat_id: str) -> str:
    """
    Recommend related papers to read next, based on the paper's topic, methods, and citations.
    """
    # Pull references/related work section from end and start
    chunks = retrieve("related work prior work references bibliography future", chat_id, top_k=6)
    kw_hits = _keyword_scan("related prior cite reference similar previous work", ocr_text, max_hits=5)
    doc_start = ocr_text[:3000]

    context_parts = [f"[Paper Start]\n{doc_start}"]
    if chunks:
        context_parts.append("[References/Related Work]\n" + "\n\n---\n\n".join(chunks))
    if kw_hits:
        context_parts.append("[Keyword Hits]\n" + "\n\n---\n\n".join(kw_hits))
    context = "\n\n".join(context_parts)

    system = (
        "You are a research librarian. Based on this paper's topic, methods, and any references mentioned, "
        "suggest 5-7 related papers a reader should explore next.\n\n"
        "For each suggestion:\n"
        "**[Paper Title]** *(Author(s), approx. year if mentioned)*\n"
        "Why relevant: [1-2 sentences explaining the connection]\n\n"
        "Rules:\n"
        "- Prioritise papers actually cited or mentioned in the text.\n"
        "- For the remaining suggestions, recommend well-known foundational or closely related work "
        "in the same research area — clearly label these as 'Suggested (not cited in paper)'.\n"
        "- Do not invent paper titles or authors. Only suggest real papers you are confident exist.\n"
        "- Group by: 1) Cited in this paper, 2) Foundational work, 3) Recent related work."
    )
    return _llm(f"Paper context:\n{context}", system=system, max_tokens=700)


# ── GENERATE PRESENTATION SLIDES ─────────────────────────────────────────────

def generate_slides(
    ocr_text: str,
    chat_id: str,
    slide_format: str = "bullets",
    detail_level: str = "normal",
    slide_count: int = 10,
    points_per_slide: int | None = None,
    point_length: str | None = None,
    query: str = "",
) -> dict:
    """
    Generate ALL slides in ONE LLM call — fast.
    Previous version called LLM once per slide (10 slides = 10 calls = very slow).
    """
    import json

    # ── Context ───────────────────────────────────────────────────────────────
    chunks    = retrieve("introduction method results conclusion contribution", chat_id, top_k=5)
    doc_start = ocr_text[:4000]
    doc_end   = ocr_text[-1500:] if len(ocr_text) > 4000 else ""
    context_parts = [f"[Paper Start]\n{doc_start}"]
    if chunks:
        context_parts.append("[Key Sections]\n" + "\n---\n".join(chunks))
    if doc_end:
        context_parts.append(f"[Paper End]\n{doc_end}")
    context = "\n\n".join(context_parts)

    # ── Slide titles ──────────────────────────────────────────────────────────
    ALL_SLIDES = [
        "Title — paper title, authors, venue/year",
        "Motivation & Problem",
        "Background / Related Work",
        "Proposed Method",
        "Architecture / Model Design",
        "Experimental Setup",
        "Key Results & Numbers",
        "Comparison with Baselines",
        "Ablation Study / Analysis",
        "Limitations",
        "Future Work",
        "Conclusion & Takeaways",
    ]
    slide_count = max(2, min(slide_count, len(ALL_SLIDES)))
    if slide_count <= 2:
        selected = [ALL_SLIDES[0], ALL_SLIDES[-1]]
    else:
        mid_needed = slide_count - 2
        mid_pool   = ALL_SLIDES[1:-1]
        step       = len(mid_pool) / mid_needed
        selected   = [ALL_SLIDES[0]] + [mid_pool[int(i * step)] for i in range(mid_needed)] + [ALL_SLIDES[-1]]

    slide_order = "\n".join(f"{i+1}. {s}" for i, s in enumerate(selected))

    # ── Exact count per slide ─────────────────────────────────────────────────
    pts = points_per_slide if points_per_slide is not None else (
        5 if detail_level == "detailed" else 2 if detail_level == "brief" else 3
    )

    # ── Per-item style instruction ────────────────────────────────────────────
    if slide_format == "paragraphs":
        if point_length == "long":
            item_rule = f"EXACTLY {pts} paragraph(s) per slide. Each paragraph: 4-6 full sentences with specific facts and numbers."
        elif point_length == "short":
            item_rule = f"EXACTLY {pts} paragraph(s) per slide. Each paragraph: 1-2 sentences."
        else:
            item_rule = f"EXACTLY {pts} paragraph(s) per slide. Each paragraph: 2-3 sentences."
        content_rule = (
            'Set "content" to [].'
            f' Write {pts} paragraph(s) into "paragraph", separated by \\n\\n.'
        )
    else:
        if point_length == "long":
            item_rule = f"EXACTLY {pts} bullet(s) per slide. Each bullet: full sentence 20-35 words."
        elif point_length == "short":
            item_rule = f"EXACTLY {pts} bullet(s) per slide. Each bullet: max 8 words, headline style."
        else:
            item_rule = f"EXACTLY {pts} bullet(s) per slide. Each bullet: 10-20 words."
        content_rule = 'Put bullets in "content" list. Set "paragraph" to "".'

    # Scale tokens: all slides in one call needs enough budget
    tokens_per_slide = pts * (120 if slide_format == "paragraphs" else 70) + 80
    max_tokens = min(max(1500, slide_count * tokens_per_slide), 7000)

    system = (
        "You are a presentation expert. Return ONLY valid JSON — no markdown fences, no extra text.\n\n"
        f"⚠️ COUNT RULE: {item_rule}\n"
        f"FORMAT RULE: {content_rule}\n\n"
        "Return this JSON structure:\n"
        "{\n"
        '  "slides": [\n'
        "    {\n"
        '      "title": "Slide title",\n'
        '      "content": ["bullet 1", "bullet 2"],\n'
        '      "paragraph": "para 1 text\\n\\npara 2 text",\n'
        '      "speaker_note": "One sentence for presenter."\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Generate EXACTLY {slide_count} slides in this order:\n{slide_order}\n\n"
        "RULES:\n"
        f"  1. Every slide gets EXACTLY {pts} items — count before writing each slide.\n"
        "  2. Use ONLY facts from the paper. No invented data.\n"
        "  3. Return valid JSON only — nothing outside the JSON object."
    )

    raw = _llm(f"Paper context:\n{context}", system=system, max_tokens=max_tokens)
    raw_clean = re.sub(r"^```[a-z]*\n?|```$", "", raw.strip(), flags=re.MULTILINE).strip()

    try:
        parsed     = json.loads(raw_clean)
        all_slides = parsed.get("slides", [])
    except Exception as e:
        logger.error(f"generate_slides JSON parse failed: {e}\nRaw: {raw_clean[:300]}")
        all_slides = []

    # ── Enforce exact count per slide (pad/trim) ──────────────────────────────
    for sl in all_slides:
        if slide_format in ("bullets", "both"):
            c = sl.get("content") or []
            while len(c) < pts:
                c.append(f"Point {len(c)+1}: see paper for details.")
            sl["content"] = c[:pts]
        if slide_format in ("paragraphs", "both"):
            paras = [p.strip() for p in (sl.get("paragraph") or "").split("\n\n") if p.strip()]
            while len(paras) < pts:
                paras.append(f"Refer to paper for additional context on {sl.get('title','this topic')}.")
            sl["paragraph"] = "\n\n".join(paras[:pts])
        if not sl.get("title"):
            sl["title"] = f"Slide {all_slides.index(sl)+1}"

    logger.info(
        f"generate_slides: format='{slide_format}' pts={pts} "
        f"count={slide_count} -> {len(all_slides)} slides (1 LLM call)"
    )
    return {
        "format":           slide_format,
        "detail":           detail_level,
        "slide_count":      slide_count,
        "points_per_slide": pts,
        "point_length":     point_length,
        "slides":           all_slides,
    }
