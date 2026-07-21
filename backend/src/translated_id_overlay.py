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
    "Group related text into sensible fields (e.g. Full Name, Sex, Date of Birth, Address, Document Number).\n"
    "For each text field, extract:\n"
    '1. "box_2d": [ymin, xmin, ymax, xmax] normalized from 0 to 1000 (where [0,0] is top-left and [1000,1000] is bottom-right).\n'
    '2. "original_text": The foreign text read from that region.\n'
    '3. "translated_text": The English translation of that field (e.g., "Name: Feng Xiangli", "Sex: Female", "Date of Birth: 24 April 1975").\n\n'
    "STRICT REQUIREMENT: Return ONLY a valid JSON array of objects like this:\n"
    "[\n"
    '  {"box_2d": [140, 280, 200, 780], "original_text": "NOM / SURNAME", "translated_text": "Name: DUPONT"},\n'
    '  {"box_2d": [220, 280, 280, 780], "original_text": "PRENOM / GIVEN NAMES", "translated_text": "Given Names: JEAN"}\n'
    "]\n"
    "Do not include markdown code blocks or conversational text. Return ONLY raw JSON."
)


def get_bold_font(size: int) -> ImageFont.FreeTypeFont:
    """Attempts to load a bold sans-serif TTF font at the given point size."""
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNS.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "arialbd.ttf",
        "arial.ttf",
    ]
    for fp in font_paths:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            continue
    return ImageFont.load_default()


def image_to_base64_png(image: Image.Image) -> str:
    """Convert PIL image to base64 PNG string."""
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


def generate_fallback_layout_boxes(
    translated_data: Dict[str, Any], img_w: int, img_h: int
) -> List[Dict[str, Any]]:
    """Generates synthetic bounding boxes on document text column if no bboxes returned."""
    items = []
    keys = [k for k, v in translated_data.items() if v]
    if not keys:
        return []

    start_y = 120
    row_h = min(75, int(700 / max(1, len(keys))))

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


def wrap_text_to_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_w: int,
) -> List[str]:
    """Wraps text into lines fitting within max_w pixels using textbbox."""
    words = text.split()
    if not words:
        return [text]

    lines = []
    cur = words[0]

    for w in words[1:]:
        test = f"{cur} {w}"
        bbox = draw.textbbox((0, 0), test, font=font)
        line_w = bbox[2] - bbox[0]

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
    Renders a two-column output canvas:
    - Left Column: Untouched original ID card image with small numbered circular source markers (1, 2, 3...)
      placed immediately beside original text boxes without covering photos or text.
    - Right Column: Dedicated "English Translation" rail containing clean, stacked, numbered translation cards.
    - Orthogonal Connector Lines: Connects each source marker to its matching rail translation card.

    Returns (canvas_png_image, count_drawn, placement_logs).
    """
    orig_w, orig_h = image.size
    rail_w = max(380, int(orig_w * 0.42))
    canvas_w = orig_w + rail_w + 40
    canvas_h = max(orig_h, 780)

    # 1. Create two-column RGBA canvas
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (248, 250, 252, 255))
    canvas.paste(image.convert("RGBA"), (0, 0))

    overlay_layer = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay_layer)

    # Color definitions
    PURPLE_MARKER = (76, 29, 149, 255)    # Dark Purple (#4C1D95)
    WHITE_TEXT = (255, 255, 255, 255)
    NAVY_TEXT = (15, 23, 42, 255)        # Dark Navy (#0F172A)
    RAIL_BG = (241, 245, 249, 255)       # Off-white container (#F1F5F9)
    RAIL_BORDER = (203, 213, 225, 255)   # Slate border (#CBD5E1)
    CARD_BG = (255, 255, 255, 255)       # Solid White Card
    CARD_BORDER = (76, 29, 149, 255)     # Purple Card Border (#4C1D95)
    LINE_COLOR = (109, 40, 217, 180)     # Purple Connector Line 70% opacity (#6D28D9)

    font_header = get_bold_font(16)
    font_marker = get_bold_font(12)
    font_card_title = get_bold_font(15)
    font_card_body = get_bold_font(14)

    # 2. Draw Translation Rail Container on Right Side
    rail_x1 = orig_w + 20
    rail_y1 = 20
    rail_x2 = canvas_w - 20
    rail_y2 = canvas_h - 20

    draw.rounded_rectangle(
        [rail_x1, rail_y1, rail_x2, rail_y2],
        radius=10,
        fill=RAIL_BG,
        outline=RAIL_BORDER,
        width=2,
    )

    # Draw Rail Header Badge
    draw.rounded_rectangle(
        [rail_x1 + 15, rail_y1 + 15, rail_x2 - 15, rail_y1 + 50],
        radius=6,
        fill=PURPLE_MARKER,
    )
    draw.text((rail_x1 + 25, rail_y1 + 22), "ENGLISH TRANSLATION RAIL", fill=WHITE_TEXT, font=font_header)

    card_y = rail_y1 + 65
    drawn_count = 0
    placements_log: List[Dict[str, Any]] = []

    # Sort text boxes top-to-bottom according to y1 coordinate
    sorted_boxes = []
    for idx, item in enumerate(text_boxes):
        translated_text = str(item.get("translated_text") or item.get("translation") or "").strip()
        coords = resolve_box_coordinates(item, orig_w, orig_h)
        if translated_text and coords:
            sorted_boxes.append((coords[1], coords, translated_text, item))

    sorted_boxes.sort(key=lambda x: x[0])  # Sort by y1 coordinate

    for marker_num, (_, coords, translated_text, orig_item) in enumerate(sorted_boxes, 1):
        x1, y1, x2, y2 = coords

        # Format numbered English text e.g. "1. Name: Feng Xiangli"
        numbered_text = f"{marker_num}. {translated_text}"

        # 3. Draw Circular Numbered Marker on Original ID Card (Left Column)
        # Position marker immediately to the right of original text bounding box
        marker_size = 18
        mx1 = min(orig_w - marker_size - 4, x2 + 6)
        my1 = max(4, y1)
        mx2 = mx1 + marker_size
        my2 = my1 + marker_size

        draw.ellipse([mx1, my1, mx2, my2], fill=PURPLE_MARKER)

        # Draw centered white marker number
        num_str = str(marker_num)
        nbox = draw.textbbox((0, 0), num_str, font=font_marker)
        nw, nh = nbox[2] - nbox[0], nbox[3] - nbox[1]
        draw.text((mx1 + (marker_size - nw) // 2, my1 + (marker_size - nh) // 2 - 1), num_str, fill=WHITE_TEXT, font=font_marker)

        # 4. Measure and Draw Translation Card in Right-hand Rail Column
        card_w = (rail_x2 - rail_x1) - 30
        card_x1 = rail_x1 + 15
        card_x2 = card_x1 + card_w

        lines = wrap_text_to_lines(draw, numbered_text, font_card_body, card_w - 20)
        card_h = max(42, len(lines) * 20 + 16)
        card_y2 = card_y + card_h

        draw.rounded_rectangle(
            [card_x1, card_y, card_x2, card_y2],
            radius=6,
            fill=CARD_BG,
            outline=CARD_BORDER,
            width=2,
        )

        # Draw English translation lines inside card
        line_y = card_y + 8
        for line in lines:
            draw.text((card_x1 + 10, line_y), line, fill=NAVY_TEXT, font=font_card_body)
            line_y += 20

        # 5. Draw Orthogonal Connector Line from Source Marker to Rail Card
        mcx, mcy = (mx1 + mx2) // 2, (my1 + my2) // 2
        card_target_y = card_y + (card_h // 2)
        elbow_x = min(orig_w + 10, mcx + 25)

        # Orthogonal elbow line: (mcx, mcy) -> (elbow_x, mcy) -> (elbow_x, card_target_y) -> (card_x1, card_target_y)
        draw.line([(mcx, mcy), (elbow_x, mcy)], fill=LINE_COLOR, width=2)
        draw.line([(elbow_x, mcy), (elbow_x, card_target_y)], fill=LINE_COLOR, width=2)
        draw.line([(elbow_x, card_target_y), (card_x1, card_target_y)], fill=LINE_COLOR, width=2)

        drawn_count += 1
        placements_log.append({
            "index": marker_num,
            "mode": "translation_rail",
            "marker_pos": [mx1, my1, mx2, my2],
            "orig_bbox": coords,
            "rail_card_bbox": [card_x1, card_y, card_x2, card_y2],
            "text": numbered_text,
        })

        logger.info(
            f"[TRANSLATION_RAIL] Marker #{marker_num} | orig_bbox={coords} | marker_pos=({mx1},{my1}) | rail_card=({card_x1},{card_y}) | text='{numbered_text[:40]}'"
        )

        card_y = card_y2 + 12

    # Composite overlay layer onto two-column canvas
    final_canvas = Image.alpha_composite(canvas, overlay_layer).convert("RGB")

    msg_summary = f"[TRANSLATION_RAIL] Rendered two-column canvas: bbox_count={len(text_boxes)}, drawn_count={drawn_count}"
    logger.info(msg_summary)
    print(msg_summary)

    return final_canvas, drawn_count, placements_log


def process_translated_id_overlay(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Performs OCR & bounding box extraction via Gemini Flash, translates text to English,
    and overlays a two-column output canvas: Left column preserves original ID card pixel-for-pixel
    with numbered circular markers; Right column presents structured English Translation Rail.
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
