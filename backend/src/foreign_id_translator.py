import base64
import json
import logging
import os
import re
from io import BytesIO
from typing import Any, Dict, List, Optional

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

# Preferred Flash models supporting generateContent & vision capabilities in order of priority
FALLBACK_FLASH_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
]


class MissingApiKeyError(ValueError):
    """Raised when GEMINI_API_KEY is missing or unconfigured."""
    pass


class InvalidApiKeyError(ValueError):
    """Raised when GEMINI_API_KEY is rejected by Google API (400/403)."""
    pass


class InvalidModelError(ValueError):
    """Raised when the requested Gemini model is not found or unsupported (404)."""
    pass


class GeminiApiFailureError(RuntimeError):
    """Raised when Gemini API request fails due to quota, network, or server errors."""
    pass


class JsonParsingError(ValueError):
    """Raised when Gemini response cannot be parsed as valid JSON."""
    pass


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


def list_available_gemini_models(api_key: str) -> List[str]:
    """
    Queries Google Gemini ModelService API to retrieve models supporting generateContent.
    Puts Flash vision models at the top of the returned list.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            raw_models = data.get("models") or []
            valid_models = []
            for m in raw_models:
                methods = m.get("supportedGenerationMethods") or []
                if "generateContent" in methods:
                    name = m.get("name", "")
                    clean_name = name.replace("models/", "")
                    if clean_name:
                        valid_models.append(clean_name)

            if valid_models:
                # Sort models so Flash models come first
                flash_first = sorted(
                    valid_models,
                    key=lambda x: (0 if "flash" in x.lower() else 1, x),
                )
                logger.info(f"Dynamically discovered {len(flash_first)} models supporting generateContent: {flash_first[:5]}")
                return flash_first
        else:
            logger.warning(f"ModelService list_models returned status {res.status_code}: {res.text[:150]}")
    except Exception as exc:
        logger.warning(f"Failed to dynamically query Gemini ModelService: {exc}")

    return FALLBACK_FLASH_MODELS


def get_model_debug_info(api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Helper function for debug endpoint listing available models and key status.
    """
    key = (api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not key or key.startswith("your-") or len(key) < 10:
        return {
            "api_key_configured": False,
            "api_key_length": len(key) if key else 0,
            "error": "GEMINI_API_KEY is not configured or is a placeholder.",
            "available_models": [],
        }

    available = list_available_gemini_models(key)
    selected_model = available[0] if available else "gemini-1.5-flash"
    return {
        "api_key_configured": True,
        "api_key_length": len(key),
        "available_models": available,
        "selected_flash_model": selected_model,
    }


def translate_foreign_id(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Analyzes an uploaded foreign ID (image or PDF) using Google Gemini Flash vision API,
    extracts details, translates to English, and returns a Python dictionary.
    """
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()

    # 1. Distinguish Error: Missing or Placeholder API Key
    if not api_key or api_key.startswith("your-") or len(api_key) < 10:
        raise MissingApiKeyError(
            "GEMINI_API_KEY is not configured in backend environment. "
            "Please set a valid GEMINI_API_KEY starting with 'AIzaSy...' under Render Dashboard -> Environment Variables."
        )

    # 2. Convert uploaded file to PIL Image and downscale for speed
    image = load_image_from_bytes(file_bytes, filename)
    base64_img = image_to_base64_jpeg(image)

    # 3. Dynamically discover available models or use fallback list
    models_to_try = list_available_gemini_models(api_key)
    if not models_to_try:
        models_to_try = FALLBACK_FLASH_MODELS

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
            logger.info(f"Calling Gemini REST API with model: {model_name}")
            res = requests.post(url, json=payload, timeout=45)

            if res.status_code == 200:
                data = res.json()
                candidates = data.get("candidates") or []
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts and parts[0].get("text"):
                        response_text = parts[0]["text"]
                        logger.info(f"Successfully received response using model: {model_name}")
                        break
            else:
                try:
                    err_json = res.json()
                except Exception:
                    err_json = {}

                err_obj = err_json.get("error") or {}
                msg = err_obj.get("message") or f"HTTP {res.status_code}: {res.text[:150]}"
                reason = (err_obj.get("details", [{}])[0] if err_obj.get("details") else {}).get("reason", "")

                logger.warning(f"Model {model_name} failed with status {res.status_code}: {msg}")
                last_error_msg = msg

                # 4. Distinguish Error: Invalid API Key vs Invalid Model
                if "API key not valid" in msg or reason == "API_KEY_INVALID" or res.status_code == 400:
                    raise InvalidApiKeyError(
                        f"GEMINI_API_KEY is invalid or rejected by Google. Details: {msg}"
                    )
                if res.status_code == 403:
                    raise InvalidApiKeyError(
                        f"GEMINI_API_KEY access forbidden or quota exceeded. Details: {msg}"
                    )
                if res.status_code == 404:
                    logger.info(f"Model {model_name} returned 404, trying next available model...")
                    continue
        except (InvalidApiKeyError, MissingApiKeyError):
            raise
        except Exception as exc:
            logger.warning(f"Request to {model_name} failed: {exc}")
            last_error_msg = str(exc)

    # 5. Distinguish Error: API Request Failure Across All Models
    if not response_text:
        raise GeminiApiFailureError(
            f"Gemini API request failed across all tested models ({', '.join(models_to_try[:4])}): {last_error_msg or 'No models accepted the request'}. "
            "Please verify your GEMINI_API_KEY on Google AI Studio."
        )

    # 6. Distinguish Error: JSON Parsing Failure
    cleaned = clean_json_response(response_text)
    try:
        translated_data = json.loads(cleaned)
        if not isinstance(translated_data, dict):
            raise JsonParsingError("Gemini response did not return a dictionary object.")
        return translated_data
    except Exception as exc:
        logger.error(f"Failed to parse Gemini JSON output: {response_text[:300]}")
        raise JsonParsingError(f"Gemini failed to return valid JSON. Response output: {response_text[:200]}") from exc
