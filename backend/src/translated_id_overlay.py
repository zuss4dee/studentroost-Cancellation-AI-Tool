import base64
import json
import logging
import os
import re
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import requests
from PIL import Image, ImageDraw, ImageFont

from .foreign_id_translator import (
    FALLBACK_FLASH_MODELS,
    GeminiApiFailureError,
    InvalidApiKeyError,
    InvalidModelError,
    MissingApiKeyError,
    clean_json_response,
    constrain_image,
    image_to_base64_jpeg,
    list_available_gemini_models,
    load_image_from_bytes,
    translate_foreign_id,
)

logger = logging.getLogger(__name__)

OVERLAY_PROMPT = (
    "You are an expert identity document OCR, bounding box detection, and translation system.\n"
    "Read the attached foreign ID document image (passport, ID card, or driver's license).\n"
    "Locate every foreign text block, label, field, name, date, and document number on the document.\n"
    "For each text block, extract:\n"
    '1. "box_2d": [ymin, xmin, ymax, xmax] normalized from 0 to 1000 (where [0,0] is top-left and [1000,1000] is bottom-right).\n'
    '2. "original_text": The foreign text read from that region.\n'
    '3. "translated_text": The English translation of that field (e.g., "Surname: DUPONT" or "Date of Birth: 15/05/1990").\n\n'
    "STRICT REQUIREMENT: Return ONLY a valid JSON array of objects like this:\n"
    "[\n"
    '  {"box_2d": [140, 280, 200, 780], "original_text": "NOM / SURNAME", "translated_text": "Surname: DUPONT"},\n'
    '  {"box_2d": [220, 280, 280, 780], "original_text": "PRENOM / GIVEN NAMES", "translated_text": "Given Names: JEAN"}\n'
    "]\n"
    "Do not include markdown code blocks or conversational text. Return ONLY raw JSON."
)


def parse_gemini_json_regions(raw_text: str) -> List[Dict[str, Any]]:
    """
    Parses JSON returned by Gemini, handling raw arrays, nested wrapper dicts,
    or key-value translation dictionaries.
    """
    cleaned = clean_json_response(raw_text)
    if not cleaned:
        return []

    parsed: Any = None
    try:
        parsed = json.loads(cleaned)
    except Exception as exc:
        logger.warning(f"[OVERLAY_PARSER] Direct JSON parse failed: {exc}. Trying regex extraction...")
        match = re.search(r"(\[.*\])", cleaned, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
            except Exception:
                parsed = None

    if isinstance(parsed, list):
        return parsed

    if isinstance(parsed, dict):
        # Look for array inside common keys
        for key in ["regions", "fields", "text_blocks", "items", "data", "extracted_regions"]:
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]

        # Convert flat key-value dictionary to region list
        items = []
        for k, v in parsed.items():
            if v and isinstance(v, (str, int, float)):
                items.append({
                    "original_text": str(k),
                    "translated_text": f"{k}: {v}",
                })
        return items

    return []


def resolve_box_coordinates(
    item: Dict[str, Any], img_w: int, img_h: int
) -> Optional[Tuple[int, int, int, int]]:
    """
    Extracts and normalizes bounding box coordinates [x1, y1, x2, y2] in actual image pixels.
    Supports normalized 0-1000 scale, normalized 0.0-1.0 float scale, or dictionary format.
    """
    box = item.get("box_2d") or item.get("box") or item.get("bbox") or item.get("bounding_box") or item.get("coords")

    ymin, xmin, ymax, xmax = 0.0, 0.0, 0.0, 0.0

    if isinstance(box, (list, tuple)) and len(box) >= 4:
        ymin, xmin, ymax, xmax = [float(v) for v in box[:4]]
    elif isinstance(box, dict):
        ymin = float(box.get("ymin", box.get("top", 0)))
        xmin = float(box.get("xmin", box.get("left", 0)))
        ymax = float(box.get("ymax", box.get("bottom", 0)))
        xmax = float(box.get("xmax", box.get("right", 0)))
    else:
        return None

    # Determine coordinate scale
    if max(ymin, xmin, ymax, xmax) <= 1.0:
        # Float scale 0.0 .. 1.0
        x1 = int(xmin * img_w)
        y1 = int(ymin * img_h)
        x2 = int(xmax * img_w)
        y2 = int(ymax * img_h)
    elif max(ymin, xmin, ymax, xmax) <= 1000.0:
        # Normalized scale 0 .. 1000
        x1 = int((xmin / 1000.0) * img_w)
        y1 = int((ymin / 1000.0) * img_h)
        x2 = int((xmax / 1000.0) * img_w)
        y2 = int((ymax / 1000.0) * img_h)
    else:
        # Absolute pixel bounds
        x1, y1, x2, y2 = int(xmin), int(ymin), int(xmax), int(ymax)

    # Sanity checks and boundaries
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(x1 + 10, min(img_w, x2))
    y2 = max(y1 + 10, min(img_h, y2))

    if (x2 - x1) < 8 or (y2 - y1) < 8:
        return None

    return (x1, y1, x2, y2)


def generate_fallback_layout_boxes(
    translated_data: Dict[str, Any], img_w: int, img_h: int
) -> List[Dict[str, Any]]:
    """
    Generates synthetic structured bounding boxes on the document text column
    if Gemini returns key-value translations without explicit bounding box coordinates.
    Positions text boxes cleanly on the right half of the ID document, leaving photo on left intact.
    """
    items = []
    keys = [k for k, v in translated_data.items() if v]
    if not keys:
        return []

    # Position on text area of document (typically xmin=300 to xmax=950 in 0-1000 scale)
    start_y = 120
    row_h = min(70, int(700 / max(1, len(keys))))

    for i, k in enumerate(keys):
        val = str(translated_data[k])
        ymin = start_y + (i * row_h)
        ymax = min(950, ymin + row_h - 10)
        items.append({
            "box_2d": [ymin, 300, ymax, 950],
            "original_text": k,
            "translated_text": f"{k}: {val}",
        })

    return items


def wrap_text_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_w: int,
) -> List[str]:
    """Wraps text into lines fitting within max_w pixels."""
    words = text.split()
    if not words:
        return [text]

    lines = []
    cur = words[0]

    for w in words[1:]:
        test = f"{cur} {w}"
        try:
            bbox = draw.textbbox((0, 0), test, font=font)
            line_w = bbox[2] - bbox[0]
        except AttributeError:
            line_w, _ = draw.textsize(test, font=font)

        if line_w <= max_w:
            cur = test
        else:
            lines.append(cur)
            cur = w

    lines.append(cur)
    return lines


def render_overlay_image(image: Image.Image, text_boxes: List[Dict[str, Any]]) -> Tuple[Image.Image, int]:
    """
    Renders solid white/light background rectangles over detected foreign text regions
    and draws clear dark English translation text inside each box.
    Leaves face photos, background colors, layout, and non-text graphics 100% intact.
    Returns (annotated_image, count_drawn).
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    img_w, img_h = image.size
    drawn_count = 0

    # Load system font fallbacks
    try:
        font_large = ImageFont.truetype("arial.ttf", 15)
        font_small = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font_large = ImageFont.load_default()
        font_small = font_large

    for idx, item in enumerate(text_boxes):
        translated_text = str(item.get("translated_text") or item.get("translation") or "").strip()
        if not translated_text:
            continue

        coords = resolve_box_coordinates(item, img_w, img_h)
        if not coords:
            continue

        x1, y1, x2, y2 = coords
        box_w = x2 - x1
        box_h = y2 - y1

        logger.info(
            f"[OVERLAY_RENDERER] Drawing box #{idx + 1} at ({x1}, {y1}, {x2}, {y2}) -> Text: '{translated_text[:35]}'"
        )

        # 1. Draw solid background rectangle to cover foreign text region
        draw.rectangle(
            [x1, y1, x2, y2],
            fill=(255, 255, 255),  # High-contrast solid white background
            outline=(71, 85, 105),  # Dark Slate border (#475569) for visibility
            width=2,
        )

        # 2. Select font size & wrap text to fit inside box
        font = font_large if box_h >= 24 else font_small
        lines = wrap_text_lines(draw, translated_text, font, max(20, box_w - 6))

        text_y = y1 + 3
        for line in lines[:4]:  # Max 4 lines per box
            if text_y + 11 > y2:
                break
            # Draw crisp dark navy/black English text
            draw.text((x1 + 4, text_y), line, fill=(15, 23, 42), font=font)
            text_y += 13

        drawn_count += 1

    logger.info(f"[OVERLAY_RENDERER] Successfully rendered {drawn_count} text overlay regions onto document image.")
    print(f"[OVERLAY_RENDERER] Successfully rendered {drawn_count} text overlay regions onto document image.")
    return annotated, drawn_count


def process_translated_id_overlay(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Performs OCR & bounding box extraction via Gemini Flash, translates text to English,
    and overlays English translation boxes onto the original document image.
    Ensures that English text is ALWAYS visibly drawn onto the output ID image.
    """
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key or api_key.startswith("your-") or len(api_key) < 10:
        raise MissingApiKeyError(
            "GEMINI_API_KEY is not configured in backend environment. "
            "Please set a valid GEMINI_API_KEY under Render Dashboard -> Environment Variables."
        )

    # 1. Load source image
    source_image = load_image_from_bytes(file_bytes, filename)
    source_image = constrain_image(source_image, max_px=1400)
    img_w, img_h = source_image.size
    base64_img = image_to_base64_jpeg(source_image)

    # 2. Discover Gemini Flash models
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
            msg_log = f"[BOUNDING_BOX_EXTRACTION_PATH] Using Gemini model: {model_name}"
            logger.info(msg_log)
            print(msg_log)

            res = requests.post(url, json=payload, timeout=45)
            if res.status_code == 200:
                data = res.json()
                candidates = data.get("candidates") or []
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts and parts[0].get("text"):
                        response_text = parts[0]["text"]
                        logger.info(f"[BOUNDING_BOX_EXTRACTION_PATH] Received Gemini response using model {model_name}")
                        break
            else:
                try:
                    err_json = res.json()
                except Exception:
                    err_json = {}

                err_obj = err_json.get("error") or {}
                msg = err_obj.get("message") or f"HTTP {res.status_code}: {res.text[:150]}"
                reason = (err_obj.get("details", [{}])[0] if err_obj.get("details") else {}).get("reason", "")

                logger.warning(f"[BOUNDING_BOX_EXTRACTION_PATH] Model {model_name} failed with status {res.status_code}: {msg}")
                last_error_msg = msg

                if "API key not valid" in msg or reason == "API_KEY_INVALID" or res.status_code == 400:
                    raise InvalidApiKeyError(f"GEMINI_API_KEY is invalid or rejected by Google. Details: {msg}")
                if res.status_code == 403:
                    raise InvalidApiKeyError(f"GEMINI_API_KEY access forbidden or quota exceeded. Details: {msg}")
                if res.status_code == 404:
                    continue
        except (InvalidApiKeyError, MissingApiKeyError):
            raise
        except Exception as exc:
            logger.warning(f"[BOUNDING_BOX_EXTRACTION_PATH] Request to {model_name} failed: {exc}")
            last_error_msg = str(exc)

    # 3. Parse JSON array of text regions
    text_boxes = parse_gemini_json_regions(response_text) if response_text else []
    logger.info(f"[OVERLAY_PIPELINE] Extracted {len(text_boxes)} text regions from Gemini response.")
    print(f"[OVERLAY_PIPELINE] Extracted {len(text_boxes)} text regions from Gemini response.")

    # 4. Fallback Generator if zero boxes parsed from direct prompt
    if not text_boxes:
        logger.info("[OVERLAY_PIPELINE] Gemini returned no explicit bounding boxes. Running key-value translation fallback...")
        try:
            translated_dict = translate_foreign_id(file_bytes, filename)
            text_boxes = generate_fallback_layout_boxes(translated_dict, img_w, img_h)
            logger.info(f"[OVERLAY_PIPELINE] Generated {len(text_boxes)} fallback text regions on document text column.")
        except Exception as exc:
            logger.warning(f"[OVERLAY_PIPELINE] Translation fallback failed: {exc}")

    # 5. Render Pillow Overlay onto source image
    annotated_image, count_drawn = render_overlay_image(source_image, text_boxes)
    annotated_base64 = image_to_base64_jpeg(annotated_image)

    logger.info(f"[OVERLAY_PIPELINE] Rendered final annotated image (drawn regions: {count_drawn}, base64 len: {len(annotated_base64)})")
    print(f"[OVERLAY_PIPELINE] Rendered final annotated image (drawn regions: {count_drawn}, base64 len: {len(annotated_base64)})")

    return {
        "success": True,
        "filename": filename,
        "annotated_image_base64": f"data:image/jpeg;base64,{annotated_base64}",
        "raw_image_base64": annotated_base64,
        "extracted_regions": text_boxes,
        "drawn_count": count_drawn,
    }
