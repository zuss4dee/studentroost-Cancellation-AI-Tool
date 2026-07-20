import base64
import json
import logging
import os
import re
from io import BytesIO
from typing import Any, Dict

import fitz  # PyMuPDF
import requests
from PIL import Image

logger = logging.getLogger(__name__)

PROMPT_TEXT = (
    "You are an expert identity document translator. Read the attached foreign ID card or passport. "
    "Extract all personal and document information, translate the field labels and the content into English. "
    "Return the result STRICTLY as a valid JSON object. Use standard keys like: "
    "'Document Type', 'First Name', 'Last Name', 'Date of Birth', 'Nationality', 'Document Number', 'Expiry Date'. "
    "Do not include markdown blocks or any conversational text, ONLY return the raw JSON object."
)


def constrain_image(image: Image.Image, max_px: int = 1400) -> Image.Image:
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


def image_to_base64_jpeg(image: Image.Image) -> str:
    """Convert PIL image to base64 JPEG string."""
    buf = BytesIO()
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    rgb.save(buf, format="JPEG", quality=85, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def translate_foreign_id(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Analyzes an uploaded foreign ID (image or PDF) using Google Gemini Flash vision REST API,
    extracts details, translates to English, and returns a Python dictionary.
    """
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key or api_key.startswith("your-") or len(api_key) < 10:
        raise ValueError(
            "GEMINI_API_KEY is not configured in backend environment. "
            "Please set a valid GEMINI_API_KEY under Render Dashboard -> Environment Variables."
        )

    # 1. Convert uploaded file to PIL Image and downscale for speed
    image = load_image_from_bytes(file_bytes, filename)
    base64_img = image_to_base64_jpeg(image)

    # 2. Try Gemini models via direct Google REST API
    models_to_try = [
        "gemini-1.5-flash",
        "gemini-2.0-flash-exp",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
    ]

    response_text = ""
    last_error_msg = ""

    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": base64_img,
                            }
                        },
                        {
                            "text": PROMPT_TEXT,
                        },
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }

        try:
            logger.info(f"Posting to Gemini REST API for model {model_name}...")
            res = requests.post(url, json=payload, timeout=45)

            if res.status_code == 200:
                data = res.json()
                candidates = data.get("candidates") or []
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts and parts[0].get("text"):
                        response_text = parts[0]["text"]
                        break
            else:
                try:
                    err_json = res.json()
                except Exception:
                    err_json = {}
                msg = (err_json.get("error") or {}).get("message") or f"HTTP {res.status_code}: {res.text[:150]}"
                logger.warning(f"Gemini REST API model {model_name} returned status {res.status_code}: {msg}")
                last_error_msg = msg

                if "API key not valid" in msg or "API_KEY_INVALID" in str(err_json):
                    raise ValueError(
                        f"GEMINI_API_KEY is invalid or rejected by Google. Details: {msg}"
                    )
        except ValueError:
            raise
        except Exception as exc:
            logger.warning(f"Request to {model_name} failed: {exc}")
            last_error_msg = str(exc)

    if not response_text:
        raise ValueError(
            f"Gemini API request failed: {last_error_msg or 'No models accepted the request'}. "
            "Please check GEMINI_API_KEY in your Render environment variables."
        )

    # 3. Clean and parse JSON response
    cleaned = clean_json_response(response_text)
    try:
        translated_data = json.loads(cleaned)
        if not isinstance(translated_data, dict):
            raise ValueError("Gemini response did not return a dictionary object.")
        return translated_data
    except Exception as exc:
        logger.error(f"Failed to parse Gemini JSON output: {response_text[:300]}")
        raise ValueError(f"Gemini failed to return valid JSON. Response output: {response_text[:200]}") from exc
