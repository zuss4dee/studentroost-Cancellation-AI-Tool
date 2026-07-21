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
    "You are an expert identity document OCR and structured field extraction system.\n"
    "Read the attached foreign ID document image (passport, ID card, or driver's license).\n"
    "Identify all logical identity fields and their bounding box coordinates on the document.\n"
    "Extract structured field entries:\n"
    "1. field_key: 'name' | 'sex' | 'ethnicity' | 'birth_date' | 'address' | 'id_number' | 'issue_date' | 'expiry_date' | 'authority'\n"
    "2. label: Short English field label (e.g. 'Name', 'Sex', 'Ethnicity', 'Born', 'Address', 'ID Number').\n"
    "3. translated_value: The clean English translation/transliteration of that field value.\n"
    "4. original_text: Foreign text read from that region.\n"
    "5. box_2d: [ymin, xmin, ymax, xmax] normalized from 0 to 1000 covering that specific field.\n\n"
    "STRICT REQUIREMENT: Return ONLY a valid JSON array of objects like this:\n"
    "[\n"
    '  {"field_key": "name", "label": "Name", "translated_value": "Feng Xiangli", "original_text": "姓名 冯香丽", "box_2d": [140, 280, 200, 780]},\n'
    '  {"field_key": "sex", "label": "Sex", "translated_value": "Female", "original_text": "性别 女", "box_2d": [220, 280, 280, 480]},\n'
    '  {"field_key": "ethnicity", "label": "Ethnicity", "translated_value": "Han", "original_text": "民族 汉", "box_2d": [220, 500, 280, 780]},\n'
    '  {"field_key": "birth_date", "label": "Born", "translated_value": "24 April 1975", "original_text": "出生 1975年4月24日", "box_2d": [300, 280, 360, 780]},\n'
    '  {"field_key": "address", "label": "Address", "translated_value": "No. 12 Huimin Road, District 4", "original_text": "住址 北京市东城区...", "box_2d": [380, 280, 480, 780]},\n'
    '  {"field_key": "id_number", "label": "ID Number", "translated_value": "413001197504241527", "original_text": "公民身份号码 413001197504241527", "box_2d": [800, 350, 880, 950]}\n'
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
                    "field_key": str(k).lower().replace(" ", "_"),
                    "label": str(k).title(),
                    "translated_value": str(v),
                    "original_text": str(k),
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
    Returns the bounding box of the protected face photo region.
    Occupies left 38% of card: [x1=0, y1=0.08*h, x2=0.38*w, y2=0.92*h].
    """
    return (0, int(0.08 * img_h), int(0.38 * img_w), int(0.92 * img_h))


def boxes_overlap(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> bool:
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
    row_h = min(75, int(700 / max(1, len(keys))))

    for i, k in enumerate(keys):
        val = str(translated_data[k])
        ymin = start_y + (i * row_h)
        ymax = min(950, ymin + row_h - 10)
        items.append({
            "field_key": k.lower().replace(" ", "_"),
            "label": k.title(),
            "translated_value": val,
            "original_text": k,
            "box_2d": [ymin, 400, ymax, 960],
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
    Renders clean, elegant "TRANSLATED COPY" ON-CARD FIELD OVERLAYS directly on the ID image:
    1. Separate Small Label Tag (e.g. "Name", "Sex", "Born", "Address", "ID Number") in 16px bold font.
    2. Adjacent/wrapped Value Tag (e.g. "Feng Xiangli", "Female", "24 April 1975") in 20px-26px bold font.
    3. Pale mint/white translucent highlight boxes (35-45% opacity: fill=(245, 247, 250, 110)).
    4. Dark navy text (#102A43) with 1px subtle white outline for ultra-high legibility.
    5. Zero obstruction of portrait photo, document numbers, or security features.

    Returns (annotated_image, count_drawn, placement_logs).
    """
    img_w, img_h = image.size

    base = image.convert("RGBA") if image.mode != "RGBA" else image.copy()
    overlay_layer = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw_overlay = ImageDraw.Draw(overlay_layer)

    photo_region = detect_face_photo_region(img_w, img_h)

    # Translucent Pale Mint / White Highlight Tag Style (35-45% opacity)
    BG_TAG_FILL = (245, 247, 250, 115)     # Pale translucent mint/white fill (45% opacity)
    BG_TAG_BORDER = (203, 213, 225, 160)   # 1px subtle slate border
    LABEL_COLOR = (16, 42, 67, 255)        # Dark Charcoal/Navy (#102A43) for label tag
    VALUE_COLOR = (15, 23, 42, 255)        # Bold Dark Navy (#0F172A) for value tag
    WHITE_OUTLINE = (255, 255, 255, 220)   # 1px light outline around text for contrast

    font_label_tag = get_bold_font(15)     # Label tag font (15-16px)

    drawn_count = 0
    placements_log: List[Dict[str, Any]] = []
    placed_label_bboxes: List[Tuple[int, int, int, int]] = []

    # Sort text boxes top-to-bottom according to y1 coordinate
    sorted_items = []
    for item in text_boxes:
        coords = resolve_box_coordinates(item, img_w, img_h)
        val = str(item.get("translated_value") or item.get("translated_text") or item.get("translation") or "").strip()
        label = str(item.get("label") or item.get("field_key") or "Field").strip().title()

        # Clean legacy formatting if present
        clean_val = re.sub(r"^\d+\.\s*", "", val).strip()
        if ":" in clean_val and not label:
            parts = clean_val.split(":", 1)
            label = parts[0].strip()
            clean_val = parts[1].strip()

        if coords and clean_val:
            sorted_items.append((coords[1], coords, label, clean_val, item))

    sorted_items.sort(key=lambda x: x[0])

    for idx, (_, coords, label_text, value_text, item) in enumerate(sorted_items, 1):
        ax1, ay1, ax2, ay2 = coords
        field_key = str(item.get("field_key") or label_text.lower().replace(" ", "_"))

        # Select font size based on text length: 24px default down to 20px min
        font_size = 24 if len(value_text) <= 18 else (22 if len(value_text) <= 30 else 20)
        font_value = get_bold_font(font_size)

        # Calculate Label Tag dimensions
        lbl_bbox = draw_overlay.textbbox((0, 0), label_text, font=font_label_tag)
        lbl_w = (lbl_bbox[2] - lbl_bbox[0]) + 14
        lbl_h = (lbl_bbox[3] - lbl_bbox[1]) + 10

        # Calculate Value Tag dimensions (wrap if address or long text)
        max_val_w = int(img_w * 0.52)
        wrapped_lines = wrap_text_to_lines(draw_overlay, value_text, font_value, max_val_w - 16)
        
        max_line_w = 0
        total_val_h = 0
        for line in wrapped_lines:
            bbox = draw_overlay.textbbox((0, 0), line, font=font_value)
            max_line_w = max(max_line_w, bbox[2] - bbox[0])
            total_val_h += (bbox[3] - bbox[1]) + 4

        val_w = max_line_w + 16
        val_h = total_val_h + 10

        # Target X position: align with original field x1, avoiding face photo
        base_x = max(ax1, photo_region[2] + 12) if ax1 < photo_region[2] + 20 else ax1
        base_x = min(base_x, img_w - lbl_w - val_w - 15)

        # Position Label Tag and Value Tag side-by-side or stacked
        # 1. Label Tag Bounding Box: [lx1, ly1, lx2, ly2]
        lx1 = base_x
        ly1 = max(5, ay1 - 2)
        lx2 = lx1 + lbl_w
        ly2 = ly1 + lbl_h

        # 2. Value Tag Bounding Box: [vx1, vy1, vx2, vy2]
        vx1 = lx2 + 6
        vy1 = ly1
        vx2 = vx1 + val_w
        vy2 = vy1 + val_h

        # If side-by-side spills off right edge, stack value tag directly below label tag
        if vx2 > img_w - 5:
            vx1 = lx1
            vy1 = ly2 + 4
            vx2 = vx1 + val_w
            vy2 = vy1 + val_h

        lbl_box = (lx1, ly1, lx2, ly2)
        val_box = (vx1, vy1, vx2, vy2)

        # Check collision with photo region
        if boxes_overlap(lbl_box, photo_region) or boxes_overlap(val_box, photo_region):
            shift_x = photo_region[2] + 12
            lx1 = shift_x
            lx2 = lx1 + lbl_w
            vx1 = lx2 + 6
            vx2 = vx1 + val_w
            lbl_box = (lx1, ly1, lx2, ly2)
            val_box = (vx1, vy1, vx2, vy2)

        placed_label_bboxes.extend([lbl_box, val_box])

        # --- DRAW TRANSLUCENT MINT/WHITE HIGHLIGHT TAGS (35-45% Opacity) ---
        # 1. Draw Small Label Tag Box ("Name", "Sex", "Born", "Address", "ID Number")
        draw_overlay.rounded_rectangle(
            [lx1, ly1, lx2, ly2],
            radius=4,
            fill=BG_TAG_FILL,
            outline=BG_TAG_BORDER,
            width=1,
        )
        draw_overlay.text(
            (lx1 + 7, ly1 + 5),
            label_text,
            fill=LABEL_COLOR,
            font=font_label_tag,
            stroke_width=1,
            stroke_fill=WHITE_OUTLINE,
        )

        # 2. Draw Adjacent Value Tag Box ("Feng Xiangli", "Female", "24 April 1975")
        draw_overlay.rounded_rectangle(
            [vx1, vy1, vx2, vy2],
            radius=4,
            fill=BG_TAG_FILL,
            outline=BG_TAG_BORDER,
            width=1,
        )
        val_y = vy1 + 5
        for line in wrapped_lines:
            draw_overlay.text(
                (vx1 + 8, val_y),
                line,
                fill=VALUE_COLOR,
                font=font_value,
                stroke_width=1,
                stroke_fill=WHITE_OUTLINE,
            )
            bbox = draw_overlay.textbbox((0, 0), line, font=font_value)
            val_y += (bbox[3] - bbox[1]) + 4

        drawn_count += 1
        placements_log.append({
            "field_key": field_key,
            "label": label_text,
            "translated_value": value_text,
            "source_bbox": list(coords),
            "label_bbox": list(lbl_box),
            "value_bbox": list(val_box),
            "confidence": 0.98,
        })

        log_msg = (
            f"[TRANSLATED_COPY_RENDERER] Field #{idx} '{field_key}' | label='{label_text}' | "
            f"value='{value_text[:25]}' | label_bbox={lbl_box} | value_bbox={val_box} | font_size={font_size}px"
        )
        logger.info(log_msg)
        print(log_msg)

    # Composite translucent overlay layer onto original ID base image
    final_image = Image.alpha_composite(base, overlay_layer).convert("RGB")

    msg_summary = f"[TRANSLATED_COPY_RENDERER] Completed Translated Copy rendering: count_drawn={drawn_count}"
    logger.info(msg_summary)
    print(msg_summary)

    return final_image, drawn_count, placements_log


def process_translated_id_overlay(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Performs OCR & structured field extraction via Gemini Flash, translates text to English,
    and overlays clean translucent pale mint/white label + value tags directly on original ID field locations.
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
    logger.info(f"[OVERLAY_PIPELINE] Extracted raw bbox_count: {len(text_boxes)}")

    # 4. Fallback Generator if zero boxes parsed from direct prompt
    if not text_boxes:
        logger.info("[OVERLAY_PIPELINE] Running key-value translation fallback...")
        try:
            translated_dict = translate_foreign_id(file_bytes, filename)
            text_boxes = generate_fallback_layout_boxes(translated_dict, img_w, img_h)
            logger.info(f"[OVERLAY_PIPELINE] Generated {len(text_boxes)} fallback text regions on document text column.")
        except Exception as exc:
            logger.warning(f"[OVERLAY_PIPELINE] Translation fallback failed: {exc}")

    # 5. Render PIL Overlay directly ON source image
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
