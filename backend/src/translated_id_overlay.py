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
    "Locate every visible printed line of text on the document.\n"
    "Group text on the same visible row into a single line entry (e.g. Name row, Sex & Ethnicity row, Date of Birth row, Address row 1, Address row 2, ID Number row).\n"
    "For each printed text line/row, extract:\n"
    '1. "box_2d": [ymin, xmin, ymax, xmax] normalized from 0 to 1000 covering that entire printed row.\n'
    '2. "original_text": The foreign text read across that entire line.\n'
    '3. "translated_text": The clean English translation for that entire printed line (e.g. "Name: Feng Xiangli", "Sex: Female | Ethnicity: Han", "Date of birth: 24 April 1975", "ID number: 413001197504241527").\n\n'
    "STRICT REQUIREMENT: Return ONLY a valid JSON array of objects like this:\n"
    "[\n"
    '  {"box_2d": [140, 280, 200, 780], "original_text": "姓名 冯香丽", "translated_text": "Name: Feng Xiangli"},\n'
    '  {"box_2d": [220, 280, 280, 780], "original_text": "性别 女 民族 汉", "translated_text": "Sex: Female | Ethnicity: Han"}\n'
    "]\n"
    "Do not include markdown code blocks or conversational text. Return ONLY raw JSON."
)


def get_bold_font(size: int) -> ImageFont.FreeTypeFont:
    """Attempts to load a bold sans-serif TTF font at the given point size (min 20px)."""
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


def cluster_text_boxes_into_lines(
    text_boxes: List[Dict[str, Any]], img_w: int, img_h: int
) -> List[Dict[str, Any]]:
    """
    Groups OCR/Gemini text detections into their original visible printed horizontal rows/lines.
    Prevents splitting one printed line into multiple labels (e.g. Sex + Ethnicity on same line).
    Produces approx 5-6 subtitle lines for standard foreign IDs.
    """
    valid_items = []
    for item in text_boxes:
        coords = resolve_box_coordinates(item, img_w, img_h)
        text = str(item.get("translated_text") or item.get("translation") or "").strip()
        orig = str(item.get("original_text") or "").strip()
        if coords and text:
            x1, y1, x2, y2 = coords
            cy = (y1 + y2) / 2.0
            h = max(10, y2 - y1)
            valid_items.append({
                "coords": coords,
                "cy": cy,
                "h": h,
                "original_text": orig,
                "translated_text": text,
                "item": item,
            })

    if not valid_items:
        return text_boxes

    # Sort items by vertical position y1
    valid_items.sort(key=lambda item: item["coords"][1])

    clusters: List[List[Dict[str, Any]]] = []
    for item in valid_items:
        placed = False
        for cluster in clusters:
            avg_cy = sum(b["cy"] for b in cluster) / len(cluster)
            avg_h = sum(b["h"] for b in cluster) / len(cluster)
            if abs(item["cy"] - avg_cy) <= max(18.0, avg_h * 0.40):
                cluster.append(item)
                placed = True
                break
        if not placed:
            clusters.append([item])

    clustered_regions: List[Dict[str, Any]] = []

    for cluster in clusters:
        cluster.sort(key=lambda b: b["coords"][0])

        line_x1 = min(b["coords"][0] for b in cluster)
        line_y1 = min(b["coords"][1] for b in cluster)
        line_x2 = max(b["coords"][2] for b in cluster)
        line_y2 = max(b["coords"][3] for b in cluster)

        orig_parts = [b["original_text"] for b in cluster if b["original_text"]]
        trans_parts = [b["translated_text"] for b in cluster if b["translated_text"]]

        combined_orig = " ".join(orig_parts) if orig_parts else ""

        unique_trans = []
        for t in trans_parts:
            # Strip any legacy numbers if present
            clean_t = re.sub(r"^\d+\.\s*", "", t).strip()
            if clean_t and clean_t not in unique_trans:
                unique_trans.append(clean_t)

        combined_trans = " | ".join(unique_trans)

        clustered_regions.append({
            "box_2d": [
                int((line_y1 / float(img_h)) * 1000.0),
                int((line_x1 / float(img_w)) * 1000.0),
                int((line_y2 / float(img_h)) * 1000.0),
                int((line_x2 / float(img_w)) * 1000.0),
            ],
            "original_text": combined_orig,
            "translated_text": combined_trans,
        })

    logger.info(f"[LINE_CLUSTERING] Clustered {len(text_boxes)} OCR items into {len(clustered_regions)} visible printed rows.")
    print(f"[LINE_CLUSTERING] Clustered {len(text_boxes)} OCR items into {len(clustered_regions)} visible printed rows.")
    return clustered_regions


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
        bbox = draw.textbbox((0, 0), test, font=font, stroke_width=3)
        line_w = bbox[2] - bbox[0]

        if line_w <= max_w:
            cur = test
        else:
            lines.append(cur)
            cur = w

    lines.append(cur)
    return lines


def calculate_subtitle_dimensions(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_allowed_w: int,
) -> Tuple[int, int, int, List[str], ImageFont.FreeTypeFont]:
    """
    Sizes subtitle font from 26px down to 20px minimum.
    Returns (label_w, label_h, font_size, wrapped_lines, font).
    """
    for font_size in range(26, 19, -1):
        font = get_bold_font(font_size)
        lines = wrap_text_to_lines(draw, text, font, max_allowed_w)

        if len(lines) <= 2:
            max_line_w = 0
            total_text_h = 0
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font, stroke_width=3)
                lw = bbox[2] - bbox[0]
                lh = bbox[3] - bbox[1]
                max_line_w = max(max_line_w, lw)
                total_text_h += lh + 4

            label_w = max_line_w
            label_h = total_text_h
            return label_w, label_h, font_size, lines, font

    font = get_bold_font(20)
    lines = wrap_text_to_lines(draw, text, font, max_allowed_w)[:2]
    max_line_w = 0
    total_text_h = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font, stroke_width=3)
        lw = bbox[2] - bbox[0]
        lh = bbox[3] - bbox[1]
        max_line_w = max(max_line_w, lw)
        total_text_h += lh + 4

    label_w = max_line_w
    label_h = total_text_h
    return label_w, label_h, 20, lines, font


def render_overlay_image(
    image: Image.Image, text_boxes: List[Dict[str, Any]]
) -> Tuple[Image.Image, int, List[Dict[str, Any]]]:
    """
    Renders clean, elegant ON-CARD TRANSLATION SUBTITLES directly on the original ID image:
    - NO numbers, NO dots, NO connector/dashed lines, NO opaque white boxes, NO right-side rail.
    - Large high-contrast Dark Navy text (#102A43) with a strong 3-4px solid white character stroke.
    - Placed 8-14px directly ABOVE or BELOW each matching printed Chinese text line.
    - Zero obstruction of portrait photo, document numbers, or security features.
    - 5-6 subtitle overlays total matching visible document lines.

    Returns (annotated_image, count_drawn, placement_logs).
    """
    img_w, img_h = image.size

    # Step 1: Cluster OCR text boxes into 5-6 visible printed rows
    clustered_lines = cluster_text_boxes_into_lines(text_boxes, img_w, img_h)

    base = image.convert("RGBA") if image.mode != "RGBA" else image.copy()
    overlay_layer = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw_overlay = ImageDraw.Draw(overlay_layer)

    photo_region = detect_face_photo_region(img_w, img_h)

    # Subtitle Palette
    TEXT_NAVY = (16, 42, 67, 255)       # Dark Navy (#102A43)
    WHITE_STROKE = (255, 255, 255, 255) # 3-4px Solid White Outline
    WHITE_SHADOW = (255, 255, 255, 180) # Subtle Soft Shadow

    drawn_count = 0
    placements_log: List[Dict[str, Any]] = []
    placed_label_bboxes: List[Tuple[int, int, int, int]] = []

    for idx, item in enumerate(clustered_lines, 1):
        translated_text = str(item.get("translated_text") or "").strip()
        # Clean any accidental legacy numbers
        clean_text = re.sub(r"^\d+\.\s*", "", translated_text).strip()

        coords = resolve_box_coordinates(item, img_w, img_h)
        if not clean_text or not coords:
            continue

        ax1, ay1, ax2, ay2 = coords

        # Max width for subtitle: up to 55% of image width
        max_allowed_w = int(img_w * 0.55)

        label_w, label_h, font_size, lines, font = calculate_subtitle_dimensions(
            draw_overlay, clean_text, max_allowed_w
        )

        placement_mode = ""
        lx1, ly1, lx2, ly2 = 0, 0, 0, 0

        # Align left edge with Chinese line start ax1, avoiding photo region
        target_x1 = max(ax1, photo_region[2] + 12) if ax1 < photo_region[2] + 20 else ax1
        target_x1 = min(target_x1, img_w - label_w - 10)

        # --- PLACEMENT CHOICE A: DIRECTLY ABOVE SOURCE LINE (8-14px) ---
        cand_y1 = ay1 - label_h - 10
        cand_y2 = cand_y1 + label_h
        cand_x1 = target_x1
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

        # --- PLACEMENT CHOICE B: DIRECTLY BELOW SOURCE LINE (8-14px) ---
        if not placement_mode:
            cand_y1 = ay2 + 8
            cand_y2 = cand_y1 + label_h
            cand_x1 = target_x1
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

        # --- PLACEMENT CHOICE C: SHIFTED ABOVE OR RIGHT ---
        if not placement_mode:
            placement_mode = "above_shifted"
            cand_y1 = max(5, ay1 - label_h - 22)
            cand_y2 = cand_y1 + label_h
            cand_x1 = max(photo_region[2] + 12, target_x1)
            cand_x2 = cand_x1 + label_w
            lx1, ly1, lx2, ly2 = cand_x1, cand_y1, cand_x2, cand_y2

        label_box = (lx1, ly1, lx2, ly2)
        placed_label_bboxes.append(label_box)

        # Render Clean Subtitle Text directly on Image
        # 1. Subtle soft shadow (+2, +2)
        # 2. Strong 3-4px solid white outline/stroke
        # 3. Solid Dark Navy text (#102A43)
        text_y = ly1
        for line in lines:
            # Soft shadow
            draw_overlay.text(
                (lx1 + 2, text_y + 2),
                line,
                fill=WHITE_SHADOW,
                font=font,
                stroke_width=4,
                stroke_fill=WHITE_SHADOW,
            )
            # Solid navy text with white stroke
            draw_overlay.text(
                (lx1, text_y),
                line,
                fill=TEXT_NAVY,
                font=font,
                stroke_width=3,
                stroke_fill=WHITE_STROKE,
            )
            bbox = draw_overlay.textbbox((0, 0), line, font=font, stroke_width=3)
            text_y += (bbox[3] - bbox[1]) + 3

        drawn_count += 1
        placements_log.append({
            "index": idx,
            "mode": placement_mode,
            "font_size": font_size,
            "lines": len(lines),
            "orig_bbox": coords,
            "label_bbox": label_box,
            "text": clean_text,
        })

        log_msg = (
            f"[SUBTITLE_RENDERER] Subtitle #{idx} | field='{clean_text[:25]}' | "
            f"placement={placement_mode.upper()} | orig_bbox={coords} | label_bbox={label_box} | "
            f"font_size={font_size}px"
        )
        logger.info(log_msg)
        print(log_msg)

    # Composite overlay layer onto original ID image
    final_image = Image.alpha_composite(base, overlay_layer).convert("RGB")

    msg_summary = (
        f"[SUBTITLE_RENDERER] Completed clean subtitle overlay rendering: original_raw_boxes={len(text_boxes)}, "
        f"line_clusters={len(clustered_lines)}, drawn_count={drawn_count}, "
        f"above_count={sum(1 for p in placements_log if 'above' in p['mode'])}, "
        f"below_count={sum(1 for p in placements_log if p['mode']=='below')}"
    )
    logger.info(msg_summary)
    print(msg_summary)

    return final_image, drawn_count, placements_log


def process_translated_id_overlay(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Performs OCR & bounding box extraction via Gemini Flash, translates text to English,
    groups OCR detections into ~5-6 printed lines, and overlays clean, elegant ON-CARD
    TRANSLATION SUBTITLES directly on the original ID image (NO numbers, NO dots, NO lines, NO white boxes).
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

    # 5. Render Subtitle Overlay directly ON source image
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
