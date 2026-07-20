import json
import logging
import os
import re
from io import BytesIO
from typing import Any, Dict

import fitz  # PyMuPDF
from PIL import Image
import google.generativeai as genai

logger = logging.getLogger(__name__)

PROMPT_TEXT = (
    "You are an expert identity document translator. Read the attached foreign ID card or passport. "
    "Extract all personal and document information, translate the field labels and the content into English. "
    "Return the result STRICTLY as a valid JSON object. Use standard keys like: "
    "'Document Type', 'First Name', 'Last Name', 'Date of Birth', 'Nationality', 'Document Number', 'Expiry Date'. "
    "Do not include markdown blocks or any conversational text, ONLY return the raw JSON object."
)


def constrain_image(image: Image.Image, max_px: int = 1600) -> Image.Image:
    """Downscale large images before sending to Gemini API to speed up processing."""
    width, height = image.size
    longest = max(width, height)
    if longest <= max_px:
        return image
    scale = max_px / longest
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


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
            return constrain_image(img)
        except Exception as exc:
            raise ValueError(f"Could not read PDF file: {exc}") from exc
    else:
        try:
            img = Image.open(BytesIO(file_bytes))
            if img.mode != "RGB":
                img = img.convert("RGB")
            return constrain_image(img)
        except Exception as exc:
            raise ValueError(f"Could not read image file: {exc}") from exc


def clean_json_response(raw_text: str) -> str:
    """Strip markdown code blocks or extra text if returned by LLM."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n?```$", "", text)
    text = text.strip()

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
            "GEMINI_API_KEY is not configured in backend environment. Please set GEMINI_API_KEY in backend/.env or Render dashboard."
        )

    # 1. Convert uploaded file to PIL Image and constrain size for speed
    image = load_image_from_bytes(file_bytes, filename)

    # 2. Configure Gemini API
    genai.configure(api_key=api_key)

    # 3. Try primary model and fallback model if needed
    models_to_try = ["gemini-1.5-flash", "gemini-2.5-flash", "gemini-1.5-pro"]
    last_error = None
    response_text = ""

    for model_name in models_to_try:
        try:
            logger.info(f"Attempting translation with model: {model_name}")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content([image, PROMPT_TEXT])
            if response and response.text:
                response_text = response.text
                break
        except Exception as exc:
            logger.warning(f"Gemini model {model_name} failed: {exc}")
            last_error = exc

    if not response_text:
        raise RuntimeError(f"Gemini API request failed across all models: {last_error}")

    # 4. Clean and parse JSON response
    cleaned = clean_json_response(response_text)
    try:
        translated_data = json.loads(cleaned)
        if not isinstance(translated_data, dict):
            raise ValueError("Gemini response did not return a dictionary object.")
        return translated_data
    except Exception as exc:
        logger.error(f"Failed to parse Gemini JSON: {response_text[:300]}")
        raise ValueError(f"Gemini failed to return valid JSON. Response output: {response_text[:200]}") from exc
