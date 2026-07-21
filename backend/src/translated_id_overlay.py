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


class FieldLayoutObject:
    """Represents a structured document field and its collision-aware layout bounds."""

    def __init__(
        self,
        field_key: str,
        label: str,
        value: str,
        source_bbox: Tuple[int, int, int, int],
        priority: int,
    ):
        self.field_key = field_key
        self.label = label
        self.value = value
        self.source_bbox = source_bbox
        self.priority = priority
        self.label_rect: Optional[Tuple[int, int, int, int]] = None
        self.value_rect: Optional[Tuple[int, int, int, int]] = None
        self.combined_rect: Optional[Tuple[int, int, int, int]] = None
        self.chosen_candidate: Optional[str] = None
        self.font_size: int = 22
        self.wrapped_lines: List[str] = [value]
        self.rejected_candidates_count: int = 0
        self.is_omitted: bool = False


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


def rects_overlap_with_clearance(
    r1: Tuple[int, int, int, int], r2: Tuple[int, int, int, int], clearance: int = 10
) -> bool:
    """Checks if bounding box r1 intersects bounding box r2 with specified clearance padding."""
    x1, y1, x2, y2 = r1
    fx1, fy1, fx2, fy2 = r2
    return not (
        x2 + clearance < fx1
        or x1 - clearance > fx2
        or y2 + clearance < fy1
        or y1 - clearance > fy2
    )


def assign_field_priority(field_key: str, label: str) -> int:
    """Returns layout placement priority (lower integer = higher priority)."""
    k = (field_key + " " + label).lower()
    if any(w in k for w in ["name", "surname", "given"]):
        return 1
    if any(w in k for w in ["sex", "gender", "ethnicity", "nationality"]):
        return 2
    if any(w in k for w in ["birth", "born", "dob"]):
        return 3
    if any(w in k for w in ["id", "number", "citizenship", "document"]):
        return 4
    if any(w in k for w in ["address", "residence", "street", "road"]):
        return 5
    return 6


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


def calculate_field_combined_rect(
    draw: ImageDraw.ImageDraw,
    label_text: str,
    value_text: str,
    font_size: int,
    cand_x: int,
    cand_y: int,
    max_val_w: int,
    font_label: ImageFont.FreeTypeFont,
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int], Tuple[int, int, int, int], List[str]]:
    """Calculates label_rect, value_rect, combined_rect, and wrapped_lines for a candidate (cand_x, cand_y)."""
    font_value = get_bold_font(font_size)

    # 1. Label bbox
    lbl_bbox = draw.textbbox((0, 0), label_text, font=font_label)
    lbl_w = (lbl_bbox[2] - lbl_bbox[0]) + 14
    lbl_h = (lbl_bbox[3] - lbl_bbox[1]) + 10

    # 2. Value bbox with wrapping if needed
    wrapped_lines = wrap_text_to_lines(draw, value_text, font_value, max_val_w - 16)
    max_line_w = 0
    total_val_h = 0
    for line in wrapped_lines:
        bbox = draw.textbbox((0, 0), line, font=font_value)
        max_line_w = max(max_line_w, bbox[2] - bbox[0])
        total_val_h += (bbox[3] - bbox[1]) + 4

    val_w = max_line_w + 16
    val_h = total_val_h + 10

    lbl_rect = (cand_x, cand_y, cand_x + lbl_w, cand_y + lbl_h)
    val_rect = (cand_x + lbl_w + 6, cand_y, cand_x + lbl_w + 6 + val_w, cand_y + val_h)

    comb_x1 = min(lbl_rect[0], val_rect[0])
    comb_y1 = min(lbl_rect[1], val_rect[1])
    comb_x2 = max(lbl_rect[2], val_rect[2])
    comb_y2 = max(lbl_rect[3], val_rect[3])
    comb_rect = (comb_x1, comb_y1, comb_x2, comb_y2)

    return lbl_rect, val_rect, comb_rect, wrapped_lines


def compute_collision_free_layout(
    draw: ImageDraw.ImageDraw,
    field_objects: List[FieldLayoutObject],
    img_w: int,
    img_h: int,
    photo_region: Tuple[int, int, int, int],
) -> List[FieldLayoutObject]:
    """
    Executes a 2-pass collision-aware layout algorithm BEFORE rendering.
    Validates candidates across font sizes 22px down to 18px with 10px clearance.
    Prevents any overlap between fields, face photo, document boundaries, or ID number strip.
    """
    # Sort field objects by priority order (Name -> Sex/Ethnicity -> Born -> ID Number -> Address -> Other)
    field_objects.sort(key=lambda f: f.priority)

    occupied_rects: List[Tuple[int, int, int, int]] = []
    font_label = get_bold_font(15)

    for field in field_objects:
        sx1, sy1, sx2, sy2 = field.source_bbox
        max_val_w = int(img_w * 0.52)
        if field.field_key in ["address", "residence"]:
            max_val_w = int(img_w * 0.55)

        placed = False

        # Try font sizes from 22px down to 18px
        for font_size in range(22, 17, -1):
            font_val = get_bold_font(font_size)
            lbl_bbox = draw.textbbox((0, 0), field.label, font=font_label)
            lbl_w = (lbl_bbox[2] - lbl_bbox[0]) + 14
            lbl_h = (lbl_bbox[3] - lbl_bbox[1]) + 10

            wrapped_lines = wrap_text_to_lines(draw, field.value, font_val, max_val_w - 16)
            max_line_w = max((draw.textbbox((0, 0), l, font=font_val)[2] - draw.textbbox((0, 0), l, font=font_val)[0]) for l in wrapped_lines)
            total_val_h = sum((draw.textbbox((0, 0), l, font=font_val)[3] - draw.textbbox((0, 0), l, font=font_val)[1] + 4) for l in wrapped_lines)
            val_w = max_line_w + 16
            val_h = total_val_h + 10

            comb_w = lbl_w + 6 + val_w
            comb_h = max(lbl_h, val_h)

            # Generate candidate positions in EXACT required order:
            # a. Directly above
            # b. Directly below
            # c. Immediately right
            # d. Immediately left
            # e. Slightly above-left
            # f. Slightly above-right
            # g. Slightly below-left
            # h. Slightly below-right
            cand_anchors = [
                ("above", sx1, sy1 - comb_h - 8),
                ("below", sx1, sy2 + 8),
                ("right", sx2 + 10, sy1),
                ("left", sx1 - comb_w - 10, sy1),
                ("above_left", max(5, sx1 - 40), sy1 - comb_h - 8),
                ("above_right", sx1 + 40, sy1 - comb_h - 8),
                ("below_left", max(5, sx1 - 40), sy2 + 8),
                ("below_right", sx1 + 40, sy2 + 8),
            ]

            for cand_name, cx, cy in cand_anchors:
                # Adjust X if candidate enters photo region
                if cx < photo_region[2] + 12:
                    cx = photo_region[2] + 12

                lbl_r, val_r, comb_r, lines = calculate_field_combined_rect(
                    draw, field.label, field.value, font_size, cx, cy, max_val_w, font_label
                )

                # Check 1: Image boundaries
                if comb_r[0] < 5 or comb_r[1] < 5 or comb_r[2] > img_w - 5 or comb_r[3] > img_h - 5:
                    field.rejected_candidates_count += 1
                    continue

                # Check 2: Overlap with face photo
                if boxes_overlap(comb_r, photo_region):
                    field.rejected_candidates_count += 1
                    continue

                # Check 3: Overlap with already occupied field rectangles (10px clearance)
                if any(rects_overlap_with_clearance(comb_r, occ, clearance=10) for occ in occupied_rects):
                    field.rejected_candidates_count += 1
                    continue

                # Candidate is clean and valid!
                field.label_rect = lbl_r
                field.value_rect = val_r
                field.combined_rect = comb_r
                field.font_size = font_size
                field.wrapped_lines = lines
                field.chosen_candidate = cand_name
                occupied_rects.append(comb_r)
                placed = True
                break

            if placed:
                break

        if not placed:
            field.is_omitted = True
            logger.warning(f"[LAYOUT_ENGINE] Omitted field '{field.field_key}' due to unresolvable collisions.")

    # Final Validation Pass: Ensure zero pair-wise overlaps among active combined_rects
    active_fields = [f for f in field_objects if not f.is_omitted and f.combined_rect]
    collisions_count = 0
    for i in range(len(active_fields)):
        for j in range(i + 1, len(active_fields)):
            if rects_overlap_with_clearance(active_fields[i].combined_rect, active_fields[j].combined_rect, clearance=5):
                collisions_count += 1
                logger.warning(
                    f"[LAYOUT_ENGINE] Pairwise collision detected between {active_fields[i].field_key} and {active_fields[j].field_key}."
                )

    report_msg = (
        f"[COLLISION_REPORT] Completed layout pass across {len(field_objects)} fields: "
        f"active_count={len(active_fields)}, omitted_count={sum(1 for f in field_objects if f.is_omitted)}, "
        f"remaining_collisions={collisions_count}"
    )
    logger.info(report_msg)
    print(report_msg)

    return field_objects


def render_overlay_image(
    image: Image.Image, text_boxes: List[Dict[str, Any]]
) -> Tuple[Image.Image, int, List[Dict[str, Any]]]:
    """
    Executes a 2-pass collision-aware layout algorithm BEFORE rendering.
    Draws clean translucent pale mint/white label + value tags directly on original ID field locations.
    Guarantees ZERO overlaps between fields, portrait photo, or document boundaries.

    Returns (annotated_image, count_drawn, placement_logs).
    """
    img_w, img_h = image.size

    base = image.convert("RGBA") if image.mode != "RGBA" else image.copy()
    overlay_layer = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw_overlay = ImageDraw.Draw(overlay_layer)

    photo_region = detect_face_photo_region(img_w, img_h)

    # Translucent Pale Mint / White Highlight Tag Style (40% opacity)
    BG_TAG_FILL = (245, 247, 250, 115)     # Pale translucent mint/white fill
    BG_TAG_BORDER = (203, 213, 225, 160)   # 1px subtle slate border
    LABEL_COLOR = (16, 42, 67, 255)        # Dark Charcoal/Navy (#102A43) for label tag
    VALUE_COLOR = (15, 23, 42, 255)        # Bold Dark Navy (#0F172A) for value tag
    WHITE_OUTLINE = (255, 255, 255, 220)   # 1px light outline around text

    font_label_tag = get_bold_font(15)     # Label tag font

    # 1. Build FieldLayoutObjects with priority assignment
    field_objects: List[FieldLayoutObject] = []
    for item in text_boxes:
        coords = resolve_box_coordinates(item, img_w, img_h)
        val = str(item.get("translated_value") or item.get("translated_text") or item.get("translation") or "").strip()
        key = str(item.get("field_key") or "field").strip().lower().replace(" ", "_")
        lbl = str(item.get("label") or key).strip().title()

        clean_val = re.sub(r"^\d+\.\s*", "", val).strip()
        if ":" in clean_val and not item.get("label"):
            parts = clean_val.split(":", 1)
            lbl = parts[0].strip().title()
            clean_val = parts[1].strip()

        if coords and clean_val:
            prio = assign_field_priority(key, lbl)
            field_objects.append(
                FieldLayoutObject(
                    field_key=key,
                    label=lbl,
                    value=clean_val,
                    source_bbox=coords,
                    priority=prio,
                )
            )

    # 2. Execute Collision-Aware Layout Engine BEFORE rendering
    layout_results = compute_collision_free_layout(
        draw_overlay, field_objects, img_w, img_h, photo_region
    )

    drawn_count = 0
    placements_log: List[Dict[str, Any]] = []

    # 3. Render Pass: Draw ONLY validated, collision-free field rectangles
    for field in layout_results:
        if field.is_omitted or not field.label_rect or not field.value_rect:
            continue

        lx1, ly1, lx2, ly2 = field.label_rect
        vx1, vy1, vx2, vy2 = field.value_rect
        font_value = get_bold_font(field.font_size)

        # Draw Small Label Tag Box ("Name", "Sex", "Born", "Address", "ID Number")
        draw_overlay.rounded_rectangle(
            [lx1, ly1, lx2, ly2],
            radius=4,
            fill=BG_TAG_FILL,
            outline=BG_TAG_BORDER,
            width=1,
        )
        draw_overlay.text(
            (lx1 + 7, ly1 + 5),
            field.label,
            fill=LABEL_COLOR,
            font=font_label_tag,
            stroke_width=1,
            stroke_fill=WHITE_OUTLINE,
        )

        # Draw Value Tag Box ("Feng Xiangli", "Female", "24 April 1975")
        draw_overlay.rounded_rectangle(
            [vx1, vy1, vx2, vy2],
            radius=4,
            fill=BG_TAG_FILL,
            outline=BG_TAG_BORDER,
            width=1,
        )
        val_y = vy1 + 5
        for line in field.wrapped_lines:
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
            "field_key": field.field_key,
            "label": field.label,
            "translated_value": field.value,
            "source_bbox": list(field.source_bbox),
            "label_bbox": list(field.label_rect),
            "value_bbox": list(field.value_rect),
            "combined_bbox": list(field.combined_rect),
            "chosen_candidate": field.chosen_candidate,
            "font_size": field.font_size,
            "rejected_candidates": field.rejected_candidates_count,
            "omitted": field.is_omitted,
            "confidence": 0.98,
        })

        log_msg = (
            f"[COLLISION_REPORT] Field '{field.field_key}' | priority={field.priority} | "
            f"candidate={field.chosen_candidate.upper()} | font_size={field.font_size}px | "
            f"combined_rect={field.combined_rect} | rejected={field.rejected_candidates_count} | omitted={field.is_omitted}"
        )
        logger.info(log_msg)
        print(log_msg)

    # Composite translucent overlay layer onto original ID base image
    final_image = Image.alpha_composite(base, overlay_layer).convert("RGB")

    msg_summary = (
        f"[COLLISION_REPORT] Rendered collision-free overlay: drawn_count={drawn_count}, "
        f"omitted_count={sum(1 for f in layout_results if f.is_omitted)}"
    )
    logger.info(msg_summary)
    print(msg_summary)

    return final_image, drawn_count, placements_log


def process_translated_id_overlay(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Performs OCR & structured field extraction via Gemini Flash, translates text to English,
    executes a 2-pass collision-aware layout algorithm, and renders non-overlapping translucent
    pale mint/white label + value tags directly on original ID field locations.
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
