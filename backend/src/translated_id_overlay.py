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


def image_to_base64_png(image: Image.Image) -> str:
    """Convert PIL image to base64 PNG string (supports RGBA alpha channels)."""
    buf = BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def parse_gemini_json_regions(raw_text: str) -> List[Dict[str, Any]]:
    """Parses JSON returned by Gemini, handling raw arrays or dict wrappers."""
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
        for key in ["regions", "fields", "text_blocks", "items", "data", "extracted_regions"]:
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]

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
    """Extracts and normalizes bounding box coordinates [x1, y1, x2, y2] in actual image pixels."""
    box = item.get("box_2d") or item.get("box") or item.get("bbox") or item.get("bounding_box") or item.get("coords")
    if not box:
        return None

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

    if max(ymin, xmin, ymax, xmax) <= 1.0:
        x1 = int(xmin * img_w)
        y1 = int(ymin * img_h)
        x2 = int(xmax * img_w)
        y2 = int(ymax * img_h)
    elif max(ymin, xmin, ymax, xmax) <= 1000.0:
        x1 = int((xmin / 1000.0) * img_w)
        y1 = int((ymin / 1000.0) * img_h)
        x2 = int((xmax / 1000.0) * img_w)
        y2 = int((ymax / 1000.0) * img_h)
    else:
        x1, y1, x2, y2 = int(xmin), int(ymin), int(xmax), int(ymax)

    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(x1 + 10, min(img_w, x2))
    y2 = max(y1 + 10, min(img_h, y2))

    if (x2 - x1) < 8 or (y2 - y1) < 8:
        return None

    return (x1, y1, x2, y2)


def detect_face_photo_region(img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """
    Returns the bounding box of the face photo area on standard ID documents.
    Typically face photo occupies left 35% of the card: [x1=0, y1=0.1*h, x2=0.38*w, y2=0.9*h].
    """
    return (0, int(0.10 * img_h), int(0.38 * img_w), int(0.90 * img_h))


def box_intersects(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> bool:
    """Checks if bounding box b1 intersects bounding box b2."""
    x1, y1, x2, y2 = b1
    fx1, fy1, fx2, fy2 = b2
    return not (x2 < fx1 or x1 > fx2 or y2 < fy1 or y1 > fy2)


def generate_fallback_layout_boxes(
    translated_data: Dict[str, Any], img_w: int, img_h: int
) -> List[Dict[str, Any]]:
    """Generates synthetic bounding boxes on document text column if no bboxes returned."""
    items = []
    keys = [k for k, v in translated_data.items() if v]
    if not keys:
        return []

    start_y = 120
    row_h = min(70, int(700 / max(1, len(keys))))

    for i, k in enumerate(keys):
        val = str(translated_data[k])
        ymin = start_y + (i * row_h)
        ymax = min(950, ymin + row_h - 10)
        items.append({
            "box_2d": [ymin, 400, ymax, 960],
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


def render_overlay_image(
    image: Image.Image, text_boxes: List[Dict[str, Any]]
) -> Tuple[Image.Image, int, List[Dict[str, Any]]]:
    """
    Renders high-legibility English text overlays onto document image using PIL RGBA layers.
    - Inline overlay: 60-70% opacity white rounded box, bold navy text, scaled & wrapped to fit.
    - Side-panel overlay + connector line: For cramped boxes or regions intersecting face photo.
    - Zero destruction: Face photo & non-text elements remain 100% untouched.

    Returns (annotated_png_image, drawn_count, placement_logs).
    """
    base = image.convert("RGBA") if image.mode != "RGBA" else image.copy()
    overlay_layer = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw_overlay = ImageDraw.Draw(overlay_layer)
    draw_base = ImageDraw.Draw(base)

    img_w, img_h = base.size
    face_region = detect_face_photo_region(img_w, img_h)

    # Color palette
    WHITE_SEMI = (255, 255, 255, 175)  # 68% opacity white background
    NAVY_TEXT = (15, 23, 42, 255)  # High contrast bold navy (#0F172A)
    BORDER_COLOR = (71, 85, 105, 200)  # Slate border (#475569)
    LINE_COLOR = (79, 70, 229, 230)  # Indigo connector line (#4F46E5)

    drawn_count = 0
    placements_log: List[Dict[str, Any]] = []
    side_panel_y = 60  # Vertical tracker for side-panel boxes

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

        # Check if region intersects face photo or is cramped (<22px height or <60px width)
        intersects_photo = box_intersects((x1, y1, x2, y2), face_region)
        is_cramped = box_h < 22 or box_w < 60

        # Determine Font Size
        font_size = max(10, min(18, int(box_h * 0.55)))
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        if not intersects_photo and not is_cramped:
            # --- MODE 1: INLINE OVERLAY (60% Semi-transparent white box) ---
            mode = "inline"
            draw_overlay.rounded_rectangle(
                [x1, y1, x2, y2],
                radius=5,
                fill=WHITE_SEMI,
                outline=BORDER_COLOR,
                width=1,
            )

            lines = wrap_text_lines(draw_overlay, translated_text, font, max(20, box_w - 6))
            text_y = y1 + 3
            for line in lines[:3]:
                if text_y + font_size > y2:
                    break
                draw_overlay.text((x1 + 4, text_y), line, fill=NAVY_TEXT, font=font)
                text_y += font_size + 2

        else:
            # --- MODE 2: SIDE-PANEL OVERLAY WITH CONNECTOR LINE ---
            mode = "side_panel"

            # Position side-panel box in open margin (e.g. right sidebar or top/bottom)
            panel_w = max(160, int(img_w * 0.32))
            panel_h = max(34, font_size * 2 + 10)
            panel_x1 = max(10, img_w - panel_w - 15)
            panel_y1 = min(img_h - panel_h - 10, side_panel_y)
            panel_x2 = panel_x1 + panel_w
            panel_y2 = panel_y1 + panel_h
            side_panel_y = panel_y2 + 12

            # 1. Draw connector line from text bbox center to side-panel
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            draw_overlay.line(
                [(center_x, center_y), (panel_x1, panel_y1 + (panel_h // 2))],
                fill=LINE_COLOR,
                width=2,
            )
            # Draw small indicator dot at source text box
            draw_overlay.ellipse(
                [center_x - 3, center_y - 3, center_x + 3, center_y + 3],
                fill=LINE_COLOR,
            )

            # 2. Draw side-panel box
            draw_overlay.rounded_rectangle(
                [panel_x1, panel_y1, panel_x2, panel_y2],
                radius=6,
                fill=WHITE_SEMI,
                outline=BORDER_COLOR,
                width=2,
            )

            lines = wrap_text_lines(draw_overlay, translated_text, font, panel_w - 10)
            text_y = panel_y1 + 4
            for line in lines[:3]:
                if text_y + font_size > panel_y2:
                    break
                draw_overlay.text((panel_x1 + 6, text_y), line, fill=NAVY_TEXT, font=font)
                text_y += font_size + 2

        drawn_count += 1
        placements_log.append({
            "index": idx + 1,
            "mode": mode,
            "font_size": font_size,
            "bbox": [x1, y1, x2, y2],
            "text": translated_text,
        })

        logger.info(
            f"[OVERLAY_RENDERER] Region #{idx + 1} [{mode.upper()}] (font_size={font_size}px, bbox=[{x1},{y1},{x2},{y2}]): '{translated_text[:35]}'"
        )

    # Composite alpha overlay layer onto base image
    final_image = Image.alpha_composite(base, overlay_layer).convert("RGB")

    msg_summary = (
        f"[OVERLAY_RENDERER] Completed overlay rendering: bbox_count={len(text_boxes)}, "
        f"drawn_count={drawn_count}, inline_count={sum(1 for p in placements_log if p['mode']=='inline')}, "
        f"side_panel_count={sum(1 for p in placements_log if p['mode']=='side_panel')}"
    )
    logger.info(msg_summary)
    print(msg_summary)

    return final_image, drawn_count, placements_log


def process_translated_id_overlay(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Performs OCR & bounding box extraction via Gemini Flash, translates text to English,
    and overlays English translation boxes onto the original document image using PIL RGBA layers.
    Saves annotated PNG image as main response.
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

    original_base64 = image_to_base64_jpeg(source_image)

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
                                "data": original_base64,
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
                logger.warning(f"[BOUNDING_BOX_EXTRACTION_PATH] Model {model_name} failed with status {res.status_code}: {msg}")
                last_error_msg = msg

                if "API key not valid" in msg or res.status_code == 400:
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
    logger.info(f"[OVERLAY_PIPELINE] Extracted bbox_count: {len(text_boxes)}")
    print(f"[OVERLAY_PIPELINE] Extracted bbox_count: {len(text_boxes)}")

    # 4. Fallback Generator if zero boxes parsed from direct prompt
    if not text_boxes:
        logger.info("[OVERLAY_PIPELINE] Running key-value translation fallback...")
        try:
            translated_dict = translate_foreign_id(file_bytes, filename)
            text_boxes = generate_fallback_layout_boxes(translated_dict, img_w, img_h)
            logger.info(f"[OVERLAY_PIPELINE] Generated {len(text_boxes)} fallback text regions on document text column.")
        except Exception as exc:
            logger.warning(f"[OVERLAY_PIPELINE] Translation fallback failed: {exc}")

    # 5. Render PIL Overlay onto source image
    annotated_image, count_drawn, placements_log = render_overlay_image(source_image, text_boxes)

    # Save output as PNG with high fidelity
    annotated_base64_png = image_to_base64_png(annotated_image)

    return {
        "success": True,
        "filename": filename,
        "annotated_image_base64": f"data:image/png;base64,{annotated_base64_png}",
        "raw_image_base64": annotated_base64_png,
        "original_image_base64": f"data:image/jpeg;base64,{original_base64}",
        "extracted_regions": text_boxes,
        "bbox_count": len(text_boxes),
        "drawn_count": count_drawn,
        "placements": placements_log,
    }
