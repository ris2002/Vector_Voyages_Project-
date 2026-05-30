import os
import logging
import base64
from io import BytesIO

import streamlit as st
import pdfplumber
from pdf2image import convert_from_bytes
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
logger = logging.getLogger(__name__)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

TEXT_THRESHOLD = 300

# Poppler path - checks env variable first, falls back to known location
POPPLER_PATH = os.getenv(
    "POPPLER_PATH",
    r"C:\Users\SPANDANA\Downloads\Release-26.02.0-0 (2)\poppler-26.02.0\Library\bin"
)


@st.cache_data(show_spinner=False)
def run_ocr(file_bytes: bytes) -> str:
    all_text = []
    page_images = convert_from_bytes(file_bytes, poppler_path="/opt/homebrew/bin")  # ← fixed

    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""

            if len(text.strip()) > TEXT_THRESHOLD:
                logger.info(f"Page {i+1}: pdfplumber ({len(text)} chars)")
                all_text.append(text)
            else:
                logger.info(f"Page {i+1}: Groq vision fallback")
                try:
                    img = page_images[i]
                    buffer = BytesIO()
                    img.save(buffer, format="JPEG")
                    b64 = base64.b64encode(buffer.getvalue()).decode()

                    response = groq_client.chat.completions.create(
                        model="meta-llama/llama-4-scout-17b-16e-instruct",
                        messages=[{
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                                },
                                {
                                    "type": "text",
                                    "text": "Extract all text from this page exactly as written."
                                }
                            ]
                        }],
                        max_tokens=2000,
                    )
                    all_text.append(response.choices[0].message.content)
                except Exception as e:
                    logger.warning(f"Groq OCR failed on page {i+1}: {e}")
                    if text.strip():
                        all_text.append(text)

    full_text = "\n\n".join(all_text)
    logger.info(f"run_ocr complete: {len(full_text)} chars")
    return full_text