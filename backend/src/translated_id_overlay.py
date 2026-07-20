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


def detect_face_photo_region(img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """
    Returns the bounding box of the protected face photo region.
    Typically occupies left 38% of card: [x1=0, y1=0.08*h, x2=0.38*w, y2=0.92*h].
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


def calculate_label_dimensions(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_allowed_w: int,
    max_lines: int = 2,
) -> Tuple[int, int, int, List[str], ImageFont.FreeTypeFont]:
    """
    Sizes font from 22px down to 15px minimum to fit within max_allowed_w.
    Returns (label_w, label_h, font_size, wrapped_lines, font).
    """
    pad_x, pad_y = 8, 5

    for font_size in range(22, 14, -1):
        font = get_bold_font(font_size)
        lines = wrap_text_to_lines(draw, text, font, max_allowed_w - (pad_x * 2))

        if len(lines) <= max_lines:
            # Calculate total width and height
            max_line_w = 0
            total_text_h = 0
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                lw = bbox[2] - bbox[0]
                lh = bbox[3] - bbox[1]
                max_line_w = max(max_line_w, lw)
                total_text_h += lh + 3

            label_w = max_line_w + (pad_x * 2)
            label_h = total_text_h + (pad_y * 2)
            return label_w, label_h, font_size, lines, font

    # Fallback to 15px font
    font = get_bold_font(15)
    lines = wrap_text_to_lines(draw, text, font, max_allowed_w - (pad_x * 2))[:max_lines]
    max_line_w = 0
    total_text_h = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        lw = bbox[2] - bbox[0]
        lh = bbox[3] - bbox[1]
        max_line_w = max(max_line_w, lw)
        total_text_h += lh + 3

    label_w = max_line_w + (pad_x * 2)
    label_h = total_text_h + (pad_y * 2)
    return label_w, label_h, 15, lines, font


def render_translation_label(
    draw_overlay: ImageDraw.ImageDraw,
    anchor_bbox: Tuple[int, int, int, int],
    translated_text: str,
    img_w: int,
    img_h: int,
    photo_region: Tuple[int, int, int, int],
    placed_label_bboxes: List[Tuple[int, int, int, int]],
    side_panel_tracker: List[int],
) -> Tuple[str, Tuple[int, int, int, int], int, int]:
    """
    Renders a compact translation label positioned ABOVE, BELOW, or SIDE_PANEL.
    Returns (placement_mode, label_bbox, font_size, line_count).
    """
    ax1, ay1, ax2, ay2 = anchor_bbox
    max_allowed_w = min(int(img_w * 0.35), 320)  # Max 35% image width (NEVER long full-width bars)

    # Determine max lines (Address gets 3, short fields get 2)
    is_address = any(word in translated_text.lower() for word in ["address", "residence", "street", "road", "district"])
    max_lines = 3 if is_address else 2

    # Measure label dimensions
    label_w, label_h, font_size, lines, font = calculate_label_dimensions(
        draw_overlay, translated_text, max_allowed_w, max_lines
    )

    # Colors: 85% opacity white background box, 2px dark navy border, solid dark navy text
    BG_WHITE_85 = (255, 255, 255, 218)  # 85% opacity white
    BORDER_NAVY = (11, 31, 58, 255)  # 2px dark navy border (#0B1F3A)
    TEXT_NAVY = (11, 31, 58, 255)  # Solid dark navy text (#0B1F3A)
    LINE_INDIGO = (79, 70, 229, 240)  # 2px connector line (#4F46E5)

    placement_mode = ""
    lx1, ly1, lx2, ly2 = 0, 0, 0, 0

    # --- PLACEMENT CHOICE A: ABOVE ---
    cand_y1 = ay1 - label_h - 6
    cand_y2 = cand_y1 + label_h
    cand_x1 = ax1
    cand_x2 = cand_x1 + label_w
    cand_box = (cand_x1, cand_y1, cand_x2, cand_y2)

    if (
        cand_y1 >= 5
        and cand_x2 <= img_w - 5
        and not boxes_overlap(cand_box, photo_region)
        and not any(boxes_overlap(cand_box, prev) for prev in placed_label_bboxes)
    ):
        placement_mode = "above"
        lx1, ly1, lx2, ly2 = cand_box

    # --- PLACEMENT CHOICE B: BELOW ---
    if not placement_mode:
        cand_y1 = ay2 + 6
        cand_y2 = cand_y1 + label_h
        cand_x1 = ax1
        cand_x2 = cand_x1 + label_w
        cand_box = (cand_x1, cand_y1, cand_x2, cand_y2)

        if (
            cand_y2 <= img_h - 5
            and cand_x2 <= img_w - 5
            and not boxes_overlap(cand_box, photo_region)
            and not any(boxes_overlap(cand_box, prev) for prev in placed_label_bboxes)
        ):
            placement_mode = "below"
            lx1, ly1, lx2, ly2 = cand_box

    # --- PLACEMENT CHOICE C: SIDE_PANEL WITH CONNECTOR LINE ---
    if not placement_mode:
        placement_mode = "side_panel"
        cur_side_y = side_panel_tracker[0]

        lx1 = max(10, img_w - label_w - 15)
        ly1 = min(img_h - label_h - 10, cur_side_y)
        lx2 = lx1 + label_w
        ly2 = ly1 + label_h
        side_panel_tracker[0] = ly2 + 10

        # Draw 2px connector line from original text center to side label
        acx = (ax1 + ax2) // 2
        acy = (ay1 + ay2) // 2
        draw_overlay.line([(acx, acy), (lx1, ly1 + (label_h // 2))], fill=LINE_INDIGO, width=2)
        draw_overlay.ellipse([acx - 3, acy - 3, acx + 3, acy + 3], fill=LINE_INDIGO)

    label_box = (lx1, ly1, lx2, ly2)
    placed_label_bboxes.append(label_box)

    # 1. Draw 85% white rounded background box with 2px dark navy border
    draw_overlay.rounded_rectangle(
        [lx1, ly1, lx2, ly2],
        radius=4,
        fill=BG_WHITE_85,
        outline=BORDER_NAVY,
        width=2,
    )

    # 2. Render solid dark navy bold English text
    text_y = ly1 + 5
    for line in lines:
        draw_overlay.text((lx1 + 8, text_y), line, fill=TEXT_NAVY, font=font)
        bbox = draw_overlay.textbbox((0, 0), line, font=font)
        lh = bbox[3] - bbox[1]
        text_y += lh + 3

    return placement_mode, label_box, font_size, len(lines)


def render_overlay_image(
    image: Image.Image, text_boxes: List[Dict[str, Any]]
) -> Tuple[Image.Image, int, List[Dict[str, Any]]]:
    """
    Renders compact English translation labels placed above, below, or side-panel.
    - Compact label boxes sized to English text (Max 35% image width).
    - Photo protection: Face area is NEVER covered.
    - High legibility: Solid dark navy text, 85% white background, 15px-22px bold font.
    - Returns (annotated_image, count_drawn, placement_logs).
    """
    base = image.convert("RGBA") if image.mode != "RGBA" else image.copy()
    overlay_layer = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw_overlay = ImageDraw.Draw(overlay_layer)

    img_w, img_h = base.size
    photo_region = detect_face_photo_region(img_w, img_h)

    drawn_count = 0
    placements_log: List[Dict[str, Any]] = []
    placed_label_bboxes: List[Tuple[int, int, int, int]] = []
    side_panel_tracker = [60]

    for idx, item in enumerate(text_boxes):
        translated_text = str(item.get("translated_text") or item.get("translation") or "").strip()
        if not translated_text:
            continue

        coords = resolve_box_coordinates(item, img_w, img_h)
        if not coords:
            continue

        placement_mode, label_box, font_size, line_count = render_translation_label(
            draw_overlay=draw_overlay,
            anchor_bbox=coords,
            translated_text=translated_text,
            img_w=img_w,
            img_h=img_h,
            photo_region=photo_region,
            placed_label_bboxes=placed_label_bboxes,
            side_panel_tracker=side_panel_tracker,
        )

        drawn_count += 1
        placements_log.append({
            "index": idx + 1,
            "mode": placement_mode,
            "font_size": font_size,
            "lines": line_count,
            "orig_bbox": coords,
            "label_bbox": label_box,
            "text": translated_text,
        })

        log_msg = (
            f"[OVERLAY_RENDERER] Label #{idx+1} | placement={placement_mode.upper()} | "
            f"orig_bbox={coords} | label_bbox={label_box} | font_size={font_size}px | "
            f"lines={line_count} | text='{translated_text[:35]}'"
        )
        logger.info(log_msg)
        print(log_msg)

    # Composite overlay layer onto base image
    final_image = Image.alpha_composite(base, overlay_layer).convert("RGB")

    msg_summary = (
        f"[OVERLAY_RENDERER] Completed overlay rendering: bbox_count={len(text_boxes)}, "
        f"drawn_count={drawn_count}, above_count={sum(1 for p in placements_log if p['mode']=='above')}, "
        f"below_count={sum(1 for p in placements_log if p['mode']=='below')}, "
        f"side_panel_count={sum(1 for p in placements_log if p['mode']=='side_panel')}"
    )
    logger.info(msg_summary)
    print(msg_summary)

    return final_image, drawn_count, placements_log


def process_translated_id_overlay(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Performs OCR & bounding box extraction via Gemini Flash, translates text to English,
    and overlays compact English translation labels onto the original document image.
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
