import json
import os
import re
from io import BytesIO
from typing import Any, Dict

import fitz  # PyMuPDF
from PIL import Image
import google.generativeai as genai

PROMPT_TEXT = (
    "You are an expert identity document translator. Read the attached foreign ID card or passport. "
    "Extract all personal and document information, translate the field labels and the content into English. "
    "Return the result STRICTLY as a valid JSON object. Use standard keys like: "
    "'Document Type', 'First Name', 'Last Name', 'Date of Birth', 'Nationality', 'Document Number', 'Expiry Date'. "
    "Do not include markdown blocks or any conversational text, ONLY return the raw JSON object."
)


def load_image_from_bytes(file_bytes: bytes, filename: str) -> Image.Image:
    """Convert uploaded file bytes (PDF or Image) into a PIL Image."""
    ext = filename.lower().split(".")[-1] if "." in filename else ""
    if ext == "pdf":
        try:
            pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
            if len(pdf_doc) == 0:
                raise ValueError("PDF file is empty.")
            page = pdf_doc[0]
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pdf_doc.close()
            return img
        except Exception as exc:
            raise ValueError(f"Could not read PDF file: {exc}") from exc
    else:
        try:
            img = Image.open(BytesIO(file_bytes))
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        except Exception as exc:
            raise ValueError(f"Could not read image file: {exc}") from exc


def clean_json_response(raw_text: str) -> str:
    """Strip markdown code blocks or extra text if returned by LLM."""
    text = raw_text.strip()
    # Remove markdown code fence if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n?```$", "", text)
    text = text.strip()

    # Extract JSON object substring if surrounded by chatter
    if not (text.startswith("{") and text.endswith("}")):
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    return text


def translate_foreign_id(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Analyzes an uploaded foreign ID (image or PDF) using gemini-1.5-flash,
    extracts details, translates to English, and returns a Python dictionary.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not configured in backend environment. Please set GEMINI_API_KEY in backend/.env."
        )

    # 1. Convert uploaded file to PIL Image
    image = load_image_from_bytes(file_bytes, filename)

    # 2. Configure Gemini API
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # 3. Call Gemini 1.5 Flash model with image & prompt
    try:
        response = model.generate_content([image, PROMPT_TEXT])
        raw_text = response.text or ""
    except Exception as exc:
        raise RuntimeError(f"Gemini API request failed: {exc}") from exc

    # 4. Clean and parse JSON response
    cleaned = clean_json_response(raw_text)
    try:
        translated_data = json.loads(cleaned)
        if not isinstance(translated_data, dict):
            raise ValueError("Gemini response did not return a dictionary object.")
        return translated_data
    except Exception as exc:
        raise ValueError(f"Gemini failed to return valid JSON. Response text: {raw_text[:200]}") from exc
