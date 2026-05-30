import os
import logging

import fitz
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

GROQ_MODEL  = "llama-3.3-70b-versatile"
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def extract_title(file_bytes: bytes, ocr_text: str, fallback: str) -> str:
    try:
        doc = fitz.open("", stream=file_bytes, filetype="pdf")
        title = doc.metadata.get("title", "").strip()
        if title:
            return title
    except Exception as e:
        logger.warning(f"PDF metadata read failed: {e}")

    prompt = (
        f"Extract ONLY the paper title from this text. "
        f"Return just the title, nothing else.\n\nText:\n{ocr_text[:500]}\n\nTitle:"
    )
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
        )
        title = response.choices[0].message.content.strip()
        if title:
            return title
    except Exception as e:
        logger.warning(f"Groq title extraction failed: {e}")

    return fallback
