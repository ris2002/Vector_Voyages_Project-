import re
import logging

logger = logging.getLogger(__name__)

INTENT_PATTERNS = {
    "summarise": [
        r"^summari[sz]e?$",
        r"^summary$",
        r"^overview$",
        r"^tl;?dr$",
        r"\bsummari[sz]e?\b",
        r"\bsummary\b",
        r"\bsynopsis\b",
        r"\bbrief (overview|description)\b",
        r"\bwhat (is|does) (this|the) paper\b",
        r"\bwhat is (it|this) about\b",
        r"\bwhat('s| is) (the )?(paper |doc |document )?about\b",
        r"\bgive me (an? )?(overview|summary|synopsis|brief|recap)\b",
        r"\btell me (what|about) (this|the) paper\b",
        r"\bdescribe (the |this )?(paper|study|article|research|document)\b",
        r"\boutline (the |this )?(paper|study|article|research)\b",
        r"\bmain (idea|point|contribution|finding|result|topic|theme)s?\b",
        r"\bkey (takeaway|takeaways|finding|findings|result|results|point|points)\b",
        r"\bwhat does this (paper|study|article|research) (do|cover|discuss|present|propose)\b",
        r"\boverall (idea|point|contribution|finding|result|theme|message)\b",
    ],
    "extract_entities": [
        r"\bwho (wrote|authored|created|published)\b",
        r"\bwho are the authors?\b",
        r"\bauthor(s)?\b",
        r"\bwhich dataset",
        r"\bwhat dataset",
        r"\bwhich model",
        r"\bwhat model",
        r"\bwhat baseline",
        r"\bwhich baseline",
        r"\bwhich metric",
        r"\bwhat metric",
        r"\bwhich benchmark",
        r"\bwhat benchmark",
        r"\bwhat (tools?|libraries?|frameworks?) (were |are )?(used|mentioned)\b",
    ],
    "compare": [
        r"\bcompar[ei]\b",          # ← FIXED: was \bcompar\b (missed the 'e')
        r"\bdifference between\b",
        r"\bversus\b",
        r"\bvs\.?\b",
        r"\bcontrast\b",
        r"\bhow does .+ differ\b",
        r"\bsimilarities? (and differences?|between)\b",
    ],
    # ── jargon FIRST — catches "difficult words", "explain term" etc ─────────
    "jargon": [
        r"\bdifficult (word|term|concept|phrase|vocabular)s?\b",
        r"\bhard (word|term|concept|phrase)s?\b",
        r"\btechnical (word|term|concept|phrase|vocabular)s?\b",
        r"\bcomplex (word|term|concept|phrase)s?\b",
        r"\bjargon\b",
        r"\bterminolog(y|ies)\b",
        r"\bvocabular(y|ies)\b",
        r"\bglossary\b",
        r"\bexplain (the |all )?(difficult|hard|complex|technical) (word|term|concept|phrase)s?\b",
        r"\b(list|give|show) (the |all )?(difficult|hard|complex|technical|unknown) (word|term|concept|phrase)s?\b",
        r"\bwhat (does|is) .+ mean\b",
        r"\bwhat do you mean by\b",
        r"\bdefine (the term|the word|the concept)\b",
        r"\bdon'?t understand (the |a |an )?(term|word|concept|phrase|acronym)\b",
        r"\bwhat does .+ stand for\b",
        r"\bwhat is .+ acronym\b",
        r"\bwords? in simple (terms?|language|words?)\b",
        r"\bsimple (terms?|language|words?) (for|of) (the |these |those )?(word|term|concept|phrase)s?\b",
    ],
    # ── simplify BEFORE explain ──────────────────────────────────────────────
    "simplify": [
        r"\bsimpl(e|ify|ified)\b",
        r"\beli5\b",
        r"\bexplain (it |this )?(like|as if).*(5|five|kid|child|beginner|layman|noob)\b",
        r"\bfor (a )?(beginner|layman|non.?expert|non.?technical|dummy|dummies|novice)\b",
        r"\bin simple (words?|terms?|language)\b",
        r"\beasy (to understand|version|explanation)\b",
        r"\bwithout jargon\b",
        r"\blayman.?s? terms?\b",
        r"\bsimple (explanation|version|summary)\b",
        r"\bbasic (explanation|overview|summary)\b",
    ],
    # ── method BEFORE explain ────────────────────────────────────────────────
    "method": [
        r"\bhow (does|did) (it|this|the paper|the model|the system|they) work\b",
        r"\bhow does .+ work\b",       # ← MOVED here from explain
        r"\bmethod(ology|ological)?\b",
        r"\bapproach (used|taken|proposed)\b",
        r"\bhow (was|were|is|are) (it|they|this) (done|built|trained|implemented|designed)\b",
        r"\barchitecture\b",
        r"\btechnique(s)?\b",
        r"\balgorithm(s)?\b",
        r"\bpipeline\b",
        r"\bhow (did|do) (they|the authors?) (do|solve|address|tackle|approach)\b",
        r"\bwhat (method|approach|technique|algorithm|model|system) (do|did|does) (they|the authors?|the paper) (use|propose|present|introduce)\b",
    ],

    "explain": [
        r"\bexplain\b",
        r"\bdefine\b",
        r"\bdefinition of\b",
        # NOTE: \bhow does .+ work\b removed — now lives in method
        r"\bwhat do you mean by\b",
        r"\bmeaning of\b",
        r"\bwhat is meant by\b",
        r"\bbreak down\b",
        r"\bwalk me through\b",
    ],
    "list_sections": [
        r"\blist (the )?sections\b",
        r"\bstructure of (the )?paper\b",
        r"\bhow is (the )?paper (organised|organized|structured)\b",
        r"\btable of contents\b",
        r"\bwhat (sections?|parts?|chapters?) does (the |this )?paper have\b",
        r"\bwhat topics (does|are) (the |this )?paper (cover|covered)\b",
    ],
    "unsolved": [
        r"\bunsolved\b",
        r"\bopen (problem|question|challenge|issue)\b",
        r"\blimitation(s)?\b",
        r"\bwhat (problem|challenge|issue)s? (remain|are left|still exist|not solved|unsolved)\b",
        r"\bfuture work\b",
        r"\bwhat.*(still|yet|remain).*(solve|done|address|explore|investigate)\b",
        r"\bwhat (gap|shortcoming|weakness|drawback|flaw)(s?)\b",
        r"\bweakness(es)?\b",
        r"\bshortcoming(s)?\b",
        r"\bdrawback(s)?\b",
        r"\bwhat (did|does) (the |this )?paper (not|fail|miss|ignore|overlook)\b",
    ],
    # ── similar_papers: added \bsuggest\b ────────────────────────────────────
    "similar_papers": [
        r"\bsimilar (paper|work|research|article|study)\b",
        r"\brelated (paper|work|research|article|study)\b",
        r"\bwhat (else|other) (should i|can i|do you) (read|suggest|recommend)\b",
        r"\brecommend (other |more )?(paper|research|article|study)\b",
        r"\bmore (like|on) (this|this topic)\b",
        r"\bother (paper|work|research) (on|about) (this|the same|similar)\b",
        r"\bsuggested reading\b",
        r"\bfurther reading\b",
        r"\bwhat to read next\b",
        r"\bfollow.?up (paper|research|work)\b",
        r"\bsuggest similar\b",        # ← NEW
        r"\bsuggest (papers?|research|articles?|studies)\b",  # ← NEW
    ],
    "slides": [
        r"\bslide(s)?\b",
        r"\bpresentation\b",
        r"\bpowerpoint\b",
        r"\bdeck\b",
        r"\bppt\b",
        r"\bcreate (a )?(slide|presentation|deck)\b",
        r"\bmake (a )?(slide|presentation|deck)\b",
        r"\bgenerate (a )?(slide|presentation|deck)\b",
        r"\bpresent (this|the) (paper|research|study)\b",
    ],
    "quiz": [
        r"\bquiz\b",
        r"\btest (me|my understanding|my knowledge)\b",
        r"\bquestion(s)? (about|on) (the |this )?paper\b",
        r"\bflash ?card(s)?\b",
        r"\bpractice question(s)?\b",
        r"\bexam question(s)?\b",
        r"\bcheck (my |the )?understanding\b",
        r"\bcan you (ask|quiz|test) me\b",
    ],
    "build_it": [
        r"\bhow (to|do i|can i|would i|could i) (build|implement|code|reproduce|create|make|develop)\b",
        r"\bimplementation (guide|steps?|plan)\b",
        r"\breproduce (the |this )?(paper|result|model|experiment|system)\b",
        r"\bpractically (build|implement|create)\b",
        r"\bstep.?by.?step (to |for )?(build|implement|code|reproduce)\b",
        r"\bhow to (build|code|implement|create|replicate) (it|this|the model|the system)\b",
        r"\bwhat (code|library|libraries|framework|tools?) (do i|would i|should i) (need|use)\b",
        r"\bbuild (this|it|the model|the system)\b",
    ],
}


def classify_intent(query: str) -> str:
    q = query.lower().strip()
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, q):
                logger.info(f"classify_intent: '{intent}' matched pattern '{pattern}'")
                return intent
    logger.info("classify_intent: no pattern matched → fallback 'qa'")
    return "qa"


def _extract_compare_terms(query: str):
    q = query.lower()
    patterns = [
        r"compare\s+(.+?)\s+(?:and|vs\.?|versus)\s+(.+)",
        r"difference\s+between\s+(.+?)\s+and\s+(.+)",
        r"(.+?)\s+(?:vs\.?|versus)\s+(.+)",
        r"how\s+(?:does|do)\s+(.+?)\s+differ\s+(?:from|to)\s+(.+)",
    ]
    for p in patterns:
        m = re.search(p, q)
        if m:
            return m.group(1).strip(), m.group(2).strip()
    parts = re.split(r"\band\b", q, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return query, ""


def _extract_explain_term(query: str) -> str:
    q = query.strip()
    patterns = [
        r"explain\s+(?:the\s+|a\s+|an\s+)?(.+?)[\?\.]*$",
        r"define\s+(?:the\s+|a\s+|an\s+)?(.+?)[\?\.]*$",
        r"meaning of\s+(.+?)[\?\.]*$",
        r"how does\s+(.+?)\s+work[\?\.]*$",
        r"walk me through\s+(.+?)[\?\.]*$",
        r"break down\s+(.+?)[\?\.]*$",
    ]
    for p in patterns:
        m = re.search(p, q, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return q


def _extract_slides_format(query: str) -> str:
    """
    Detect whether the user wants bullet-point slides or paragraph-style slides.
    Returns 'bullets' (default), 'paragraphs', or 'both'.
    """
    q = query.lower()
    paragraph_signals = [
        r"\bparagraph(s)?\b",
        r"\bpara(s)?\b",
        r"\bin prose\b",
        r"\bfull sentence(s)?\b",
        r"\bwritten (out|form)\b",
        r"\bnarrative\b",
        r"\bno bullet(s)?\b",
        r"\bwithout bullet(s)?\b",
        r"\bnot (in )?bullet(s)?\b",
        r"\blong.?form\b",
    ]
    bullet_signals = [
        r"\bbullet(s| point(s)?)?\b",
        r"\bin points\b",
        r"\bshort (point|line|item)(s)?\b",
        r"\bconcise\b",
        r"\bbrief point(s)?\b",
    ]

    has_para   = any(re.search(p, q) for p in paragraph_signals)
    has_bullet = any(re.search(p, q) for p in bullet_signals)

    if has_para and has_bullet:
        logger.info("_extract_slides_format: detected 'both'")
        return "both"
    if has_para:
        logger.info("_extract_slides_format: detected 'paragraphs'")
        return "paragraphs"
    if has_bullet:
        logger.info("_extract_slides_format: detected 'bullets'")
        return "bullets"
    logger.info("_extract_slides_format: no signal → default 'bullets'")
    return "bullets"


def _extract_slide_count(query: str) -> int | None:   # ← return type: int | None
    q = query.lower()
    word_to_num = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    }
    m = re.search(r'\b(\d+)\s*slides?\b', q)
    if m:
        n = int(m.group(1))
        if 2 <= n <= 30:
            logger.info(f"_extract_slide_count: found digit count {n}")
            return n
    for word, num in word_to_num.items():
        if re.search(rf'\b{word}\s+slides?\b', q):
            logger.info(f"_extract_slide_count: found word count {num}")
            return num
    logger.info("_extract_slide_count: no count found → None")
    return None   # ← CHANGED from 10 to None

def _extract_detail_level(query: str) -> str:
    """
    Detect how much detail the user wants per slide.
    Only used when points_per_slide is NOT explicitly given.
    Returns 'brief', 'normal' (default), or 'detailed'.
    """
    q = query.lower()
    detailed_signals = [
        r"\bdetail(ed|s)?\b",
        r"\bin depth\b",
        r"\bin-depth\b",
        r"\bmore info(rmation)?\b",
        r"\bmore content\b",
        r"\bcomprehensive\b",
        r"\bexpand(ed)?\b",
        r"\belaborate\b",
        r"\bexhaustive\b",
        r"\beverything\b",
        r"\ball (the )?info\b",
    ]
    brief_signals = [
        r"\bbrief\b",
        r"\bquick\b",
        r"\bminimal\b",
        r"\bkept? short\b",
        r"\bfew (words?|points?)\b",
    ]

    if any(re.search(p, q) for p in detailed_signals):
        logger.info("_extract_detail_level: detected 'detailed'")
        return "detailed"
    if any(re.search(p, q) for p in brief_signals):
        logger.info("_extract_detail_level: detected 'brief'")
        return "brief"
    logger.info("_extract_detail_level: default 'normal'")
    return "normal"


def _extract_points_per_slide(query: str) -> int | None:
    """
    Extract how many bullet points or paragraphs the user wants per slide.

    Handles all natural phrasings such as:
      "6 slides with 6 points each"
      "each slide should have 4 bullets"
      "3 points per slide"
      "2 paragraphs per slide"
      "slides with 5 bullet points each"
      "give me 5 slides each having 8 points"
      "10 slides 3 points"
      "five slides with three bullets each"

    Returns an int (1–15) if an explicit count is found, else None.
    """
    q = query.lower()

    word_to_num = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }

    content_word = r"(?:points?|bullets?|bullet\s*points?|paragraphs?|paras?|items?|lines?)"

    # Digit-based patterns (most specific first)
    digit_patterns = [
        # "6 points each" / "6 points per slide" / "6 points per"
        rf'\b(\d+)\s*{content_word}\s*(?:each|per slide|per)\b',
        # "each slide * 4 points" / "each slide should have 4 bullets"
        rf'\beach\s+slide\s+(?:\w+\s+){{0,5}}(\d+)\s*{content_word}\b',
        # "4 points on each" / "4 points each slide"
        rf'\b(\d+)\s*{content_word}\s+(?:on\s+)?(?:each|every)\b',
        # "with 5 points" / "having 5 points" / "of 5 points" / "contain 5 bullets"
        rf'\b(?:with|having|of|contain|include|give)\s+(\d+)\s*{content_word}\b',
        # "slides with 5 each"
        r'\bslides?\s+with\s+(\d+)\s+each\b',
        # plain "5 points" anywhere (least specific, run last)
        rf'\b(\d+)\s*{content_word}\b',
    ]

    for pattern in digit_patterns:
        m = re.search(pattern, q)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 15:
                logger.info(f"_extract_points_per_slide: found {n} via pattern '{pattern}'")
                return n

    # Word-number versions of the same patterns
    for word, num in word_to_num.items():
        word_patterns = [
            rf'\b{word}\s*{content_word}\s*(?:each|per slide|per)\b',
            rf'\beach\s+slide\s+(?:\w+\s+){{0,5}}{word}\s*{content_word}\b',
            rf'\b{word}\s*{content_word}\s+(?:on\s+)?(?:each|every)\b',
            rf'\b(?:with|having|of|contain|include|give)\s+{word}\s*{content_word}\b',
            rf'\b{word}\s*{content_word}\b',
        ]
        for pattern in word_patterns:
            if re.search(pattern, q):
                logger.info(f"_extract_points_per_slide: found {num} (word) via pattern '{pattern}'")
                return num

    logger.info("_extract_points_per_slide: no explicit count → None")
    return None


def _extract_point_length(query: str) -> str | None:
    """
    Detect whether the user wants long/detailed or short/concise individual points.
    Returns 'long', 'short', or None.
    """
    q = query.lower()

    long_signals = [
        r"\blong\s*(points?|bullets?|sentences?|paragraphs?)?\b",
        r"\bdetailed\s*(points?|bullets?|sentences?)?\b",
        r"\belaborate\s*(points?|bullets?)?\b",
        r"\bin.?depth\s*(points?|bullets?)?\b",
        r"\bexpanded?\s*(points?|bullets?)?\b",
        r"\bfull\s*(sentences?|points?|bullets?)?\b",
        r"\bcomprehensive\s*(points?|bullets?)?\b",
        r"\bdescriptive\s*(points?|bullets?)?\b",
    ]
    short_signals = [
        r"\bshort\s*(points?|bullets?|sentences?|paragraphs?)?\b",
        r"\bconcise\s*(points?|bullets?)?\b",
        r"\bbrief\s*(points?|bullets?)?\b",
        r"\bone.?liner(s)?\b",
        r"\bshort and sweet\b",
        r"\bto the point\b",
        r"\bminimal\s*(points?|bullets?)?\b",
        r"\bquick\s*(points?|bullets?)?\b",
        r"\bsnappy\b",
    ]

    if any(re.search(p, q) for p in long_signals):
        logger.info("_extract_point_length: detected 'long'")
        return "long"
    if any(re.search(p, q) for p in short_signals):
        logger.info("_extract_point_length: detected 'short'")
        return "short"
    return None


def _extract_jargon_term(query: str) -> str:
    """
    Extract the specific term a user is asking about.
    For bulk requests (difficult words, glossary etc), returns the full query
    so define_jargon can detect it as a bulk request.
    """
    import re as _re
    q = query.strip()

    # Bulk signals — return full query so define_jargon handles it
    bulk_signals = [
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
    ]
    if any(_re.search(p, q.lower()) for p in bulk_signals):
        logger.info(f"_extract_jargon_term: bulk request detected — passing full query")
        return q  # define_jargon will handle it as bulk

    # Single term extraction
    patterns = [
        r"what (?:does|is) (.+?) mean",
        r"define (?:the (?:term|word|concept) )?(.+?)[\?\.]*$",
        r"what does (.+?) stand for",
        r"don'?t understand (?:the |a |an )?(.+?)[\?\.]*$",
        r"what is (.+?) (?:in this paper|here|used)",
    ]
    for p in patterns:
        m = re.search(p, q, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # Fallback: strip common lead words
    cleaned = re.sub(
        r"^(what is|what does|what are|define|explain|meaning of|what's)\s+",
        "", q, flags=re.IGNORECASE
    ).strip(" ?.")
    return cleaned if cleaned else q



def _extract_quiz_count(query: str) -> int:
    """Extract how many quiz questions the user wants. Default 5."""
    q = query.lower()
    word_to_num = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    }
    # digit: "give me 8 questions" / "9 quiz questions" / "10 mcqs"
    m = re.search(r"\b(\d+)\s*(?:quiz\s*)?(?:question|mcq|q)s?\b", q)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 20:
            return n
    # word: "give me ten questions"
    for word, num in word_to_num.items():
        if re.search(rf"\b{word}\s+(?:quiz\s*)?(?:question|mcq|q)s?\b", q):
            return num
    return 5  # default


def run_agent(
    query: str,
    chat_id: str,
    history: list[dict],
    ocr_text: str = "",
) -> str:
    from .tools import (
        summarise_paper,
        answer_from_document,
        answer_complex_question,
        extract_entities,
        compare_concepts,
        explain_concept,
        list_sections,
        simplify_paper,
        define_jargon,
        find_unsolved_problems,
        explain_method,
        generate_quiz,
        build_it_guide,
        suggest_similar_papers,
        generate_slides,
    )

    intent = classify_intent(query)
    logger.info(f"run_agent: intent='{intent}' | query='{query[:80]}'")

    def _need_ocr(intent_name: str) -> bool:
        if not ocr_text:
            logger.warning(f"run_agent: ocr_text missing for intent '{intent_name}'")
            return False
        return True

    if intent == "summarise":
        if not _need_ocr(intent): return "OCR text not available — please re-upload the PDF."
        return summarise_paper(ocr_text)

    elif intent == "extract_entities":
        if not _need_ocr(intent): return "OCR text not available — please re-upload the PDF."
        return extract_entities(ocr_text)

    elif intent == "list_sections":
        if not _need_ocr(intent): return "OCR text not available — please re-upload the PDF."
        return list_sections(ocr_text)

    elif intent == "compare":
        term_a, term_b = _extract_compare_terms(query)
        if not term_b:
            return answer_from_document(query, chat_id, history, ocr_text=ocr_text)
        return compare_concepts(term_a, term_b, chat_id, ocr_text=ocr_text)

    elif intent == "explain":
        term = _extract_explain_term(query)
        return explain_concept(term, chat_id, ocr_text=ocr_text)

    elif intent == "simplify":
        if not _need_ocr(intent): return "OCR text not available — please re-upload the PDF."
        return simplify_paper(ocr_text)

    elif intent == "jargon":
        term = _extract_jargon_term(query)
        return define_jargon(term, chat_id, ocr_text=ocr_text)

    elif intent == "unsolved":
        if not _need_ocr(intent): return "OCR text not available — please re-upload the PDF."
        return find_unsolved_problems(ocr_text, chat_id)

    elif intent == "method":
        if not _need_ocr(intent): return "OCR text not available — please re-upload the PDF."
        return explain_method(ocr_text, chat_id)

    elif intent == "quiz":
        if not _need_ocr(intent): return "OCR text not available — please re-upload the PDF."
        num_q = _extract_quiz_count(query)
        return generate_quiz(ocr_text, quiz_count=num_q)

    elif intent == "build_it":
        if not _need_ocr(intent): return "OCR text not available — please re-upload the PDF."
        return build_it_guide(ocr_text, chat_id)

    elif intent == "similar_papers":
        if not _need_ocr(intent): return "OCR text not available — please re-upload the PDF."
        return suggest_similar_papers(ocr_text, chat_id)

    elif intent == "slides":
        if not _need_ocr(intent): return {"error": "OCR text not available — please re-upload the PDF."}

        slide_format     = _extract_slides_format(query)
        detail_level     = _extract_detail_level(query)
        slide_count      = _extract_slide_count(query) or 10
        points_per_slide = _extract_points_per_slide(query)
        point_length     = _extract_point_length(query)

        logger.info(
            f"run_agent slides → format={slide_format}, detail={detail_level}, "
            f"count={slide_count}, points_per_slide={points_per_slide}, point_length={point_length}"
        )

        return generate_slides(
            ocr_text,
            chat_id,
            slide_format=slide_format,
            detail_level=detail_level,
            slide_count=slide_count,
            points_per_slide=points_per_slide,
            point_length=point_length,
            query=query,
        )

    elif intent == "qa" and len(query.split()) > 12:
        return answer_complex_question(query, chat_id, history, ocr_text=ocr_text)

    else:
        return answer_from_document(query, chat_id, history, ocr_text=ocr_text)
