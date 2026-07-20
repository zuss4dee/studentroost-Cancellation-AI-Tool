import base64
import json
import logging
import os
import re
from io import BytesIO
from typing import Any, Dict, List, Tuple

import fitz  # PyMuPDF
import requests
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

OVERLAY_PROMPT = (
    "You are an expert identity document OCR and translation system.\n"
    "Analyze the attached foreign ID document image (passport, ID card, or driver's license).\n"
    "Detect all foreign text labels, names, dates, numbers, and text regions.\n"
    "For each text region, return:\n"
    '1. "box_2d": [ymin, xmin, ymax, xmax] coordinates normalized from 0 to 1000 (where top-left is [0, 0] and bottom-right is [1000, 1000]).\n'
    '2. "original_text": The foreign text detected in that region.\n'
    '3. "translated_text": The English translation or English equivalent of that field.\n\n'
    "STRICT REQUIREMENT: Return a valid JSON array of objects.\n"
    "Example:\n"
    "[\n"
    '  {"box_2d": [120, 250, 180, 750], "original_text": "NOM / SURNAME", "translated_text": "Surname: DUPONT"}\n'
    "]\n"
    "Do not include markdown blocks or conversational text, ONLY return the raw JSON array."
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


def image_to_base64_jpeg(image: Image.Image) -> str:
    """Convert PIL image to base64 JPEG string."""
    buf = BytesIO()
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    rgb.save(buf, format="JPEG", quality=88, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def clean_json_response(raw_text: str) -> str:
    """Strip markdown fences if present."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n?```$", "", text)
    text = text.strip()

    if not (text.startswith("[") and text.endswith("]")):
        match = re.search(r"(\[.*\])", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    return text


def wrap_text_to_fit(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> List[str]:
    """Wrap text so each line fits within max_width pixels."""
    words = text.split()
    if not words:
        return [text]

    lines = []
    current_line = words[0]

    for word in words[1:]:
        test_line = f"{current_line} {word}"
        try:
            bbox = draw.textbbox((0, 0), test_line, font=font)
            line_w = bbox[2] - bbox[0]
        except AttributeError:
            line_w, _ = draw.textsize(test_line, font=font)

        if line_w <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word

    lines.append(current_line)
    return lines


def render_overlay_image(image: Image.Image, text_boxes: List[Dict[str, Any]]) -> Image.Image:
    """
    Renders clean white background boxes over detected foreign text regions
    and draws the English translation text inside each bounding box.
    Leaves non-text regions (photos, backgrounds, layout) 100% untouched.
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    img_w, img_h = image.size

    # Load font or default
    try:
        font_large = ImageFont.truetype("arial.ttf", 14)
        font_small = ImageFont.truetype("arial.ttf", 10)
    except Exception:
        font_large = ImageFont.load_default()
        font_small = font_large

    for item in text_boxes:
        box_2d = item.get("box_2d") or item.get("box") or item.get("bounding_box")
        translated_text = str(item.get("translated_text") or item.get("translation") or "").strip()
        if not box_2d or len(box_2d) < 4 or not translated_text:
            continue

        ymin, xmin, ymax, xmax = box_2d[:4]

        # Convert normalized 0-1000 coordinates to actual image pixel coordinates
        x1 = max(0, int(xmin / 1000.0 * img_w))
        y1 = max(0, int(ymin / 1000.0 * img_h))
        x2 = min(img_w, int(xmax / 1000.0 * img_w))
        y2 = min(img_h, int(ymax / 1000.0 * img_h))

        box_w = max(16, x2 - x1)
        box_h = max(12, y2 - y1)

        # 1. Draw clean background rectangle over foreign text area
        draw.rectangle(
            [x1, y1, x2, y2],
            fill=(255, 255, 255),  # Solid white overlay
            outline=(203, 213, 225),  # Subtle Slate border
            width=1,
        )

        # 2. Fit and render English translated text inside bounding box
        font = font_large if box_h > 24 else font_small
        lines = wrap_text_to_fit(draw, translated_text, font, box_w - 4)

        text_y = y1 + 2
        for line in lines[:3]:  # Max 3 lines inside box
            if text_y + 10 > y2:
                break
            draw.text((x1 + 3, text_y), line, fill=(15, 23, 42), font=font)
            text_y += 12

    return annotated


def process_translated_id_overlay(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Performs OCR & bounding box extraction via Gemini Flash, translates text to English,
    and overlays English translation boxes onto the original document image.
    """
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key or api_key.startswith("your-") or len(api_key) < 10:
        raise ValueError(
            "GEMINI_API_KEY is missing or invalid. Please set GEMINI_API_KEY under Render Dashboard -> Environment Variables."
        )

    # 1. Load source image
    source_image = load_image_from_bytes(file_bytes, filename)
    base64_img = image_to_base64_jpeg(source_image)

    # 2. Call Gemini Flash API for Bounding Box + Translation
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
                            "text": OVERLAY_PROMPT,
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
            logger.info(f"Detecting text regions & bounding boxes with model {model_name}...")
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
                logger.warning(f"Model {model_name} returned status {res.status_code}: {msg}")
                last_error_msg = msg
        except Exception as exc:
            logger.warning(f"Overlay request to {model_name} failed: {exc}")
            last_error_msg = str(exc)

    if not response_text:
        raise RuntimeError(
            f"Gemini Vision bounding box extraction failed: {last_error_msg or 'No models accepted the request'}"
        )

    # 3. Parse JSON array of text regions & bounding boxes
    cleaned = clean_json_response(response_text)
    try:
        text_boxes = json.loads(cleaned)
        if not isinstance(text_boxes, list):
            text_boxes = []
    except Exception as exc:
        logger.error(f"Failed to parse text bounding boxes JSON: {response_text[:300]}")
        text_boxes = []

    # 4. Render Pillow Overlay onto source image
    annotated_image = render_overlay_image(source_image, text_boxes)
    annotated_base64 = image_to_base64_jpeg(annotated_image)

    return {
        "success": True,
        "filename": filename,
        "annotated_image_base64": f"data:image/jpeg;base64,{annotated_base64}",
        "raw_image_base64": annotated_base64,
        "extracted_regions": text_boxes,
    }
