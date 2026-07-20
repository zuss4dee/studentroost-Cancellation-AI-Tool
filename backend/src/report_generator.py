from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def format_value(value: Any) -> str:
    """Format dictionary values into clean readable strings."""
    if value is None:
        return "N/A"
    if isinstance(value, (dict, list)):
        import json
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def flatten_dictionary(d: Dict[str, Any], prefix: str = "") -> List[Tuple[str, str]]:
    """Flatten nested dicts/lists into key-value tuples for 2-column table display."""
    items: List[Tuple[str, str]] = []
    for key, val in d.items():
        full_key = f"{prefix}{key}" if prefix else str(key)
        if isinstance(val, dict):
            items.extend(flatten_dictionary(val, prefix=f"{full_key} -> "))
        elif isinstance(val, list):
            if all(isinstance(item, dict) for item in val):
                for idx, sub_item in enumerate(val, 1):
                    items.extend(flatten_dictionary(sub_item, prefix=f"{full_key} #{idx} -> "))
            else:
                formatted_list = ", ".join(format_value(item) for item in val)
                items.append((full_key, formatted_list))
        else:
            items.append((full_key, format_value(val)))
    return items


def generate_translated_id_pdf(translated_data: Dict[str, Any], source_filename: str = "") -> BytesIO:
    """
    Generates a PDF document summary of translated foreign ID fields using ReportLab.
    
    Features:
    - Title: Translated Identity Document Summary
    - 2-Column Table (Field Name, Translated Detail)
    - Dark blue header row with white bold text
    - Alternating row backgrounds and clear grid lines
    - Helvetica font
    - Returns BytesIO buffer containing PDF binary data
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
        title="Translated Identity Document Summary",
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1E3A8A"),  # Dark Blue
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748B"),  # Muted grey
        spaceAfter=14,
    )

    header_cell_style = ParagraphStyle(
        "HeaderCell",
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.white,
    )

    key_cell_style = ParagraphStyle(
        "KeyCell",
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#1E293B"),
    )

    val_cell_style = ParagraphStyle(
        "ValCell",
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#334155"),
    )

    elements = []

    # 1. Header Section
    elements.append(Paragraph("Translated Identity Document Summary", title_style))
    current_time = datetime.now().strftime("%B %d, %Y - %H:%M:%S")
    sub_text = f"Source Document: <b>{source_filename or 'Uploaded ID'}</b> | Extracted & Translated: {current_time}"
    elements.append(Paragraph(sub_text, subtitle_style))
    elements.append(Spacer(1, 10))

    # 2. Build 2-Column Table Data
    table_data = [
        [
            Paragraph("Field Name", header_cell_style),
            Paragraph("Translated Detail", header_cell_style),
        ]
    ]

    flat_items = flatten_dictionary(translated_data)
    if not flat_items:
        flat_items = [("Status", "No fields were extracted.")]

    for key, val in flat_items:
        table_data.append([
            Paragraph(str(key), key_cell_style),
            Paragraph(str(val), val_cell_style),
        ])

    # Printable width = 8.5" * 72 - 72 (margins) = 540 pt
    # Col 1 (Field Name): 180 pt, Col 2 (Translated Detail): 360 pt
    table = Table(table_data, colWidths=[180, 360])

    # 3. Table Styling
    ts = [
        ("BACKGROUND", (0, 0), (1, 0), colors.HexColor("#1E3A8A")),  # Dark Blue header
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),  # Grid lines
    ]

    # Alternating row colors
    for row in range(1, len(table_data)):
        if row % 2 == 0:
            ts.append(("BACKGROUND", (0, row), (1, row), colors.HexColor("#F8FAFC")))
        else:
            ts.append(("BACKGROUND", (0, row), (1, row), colors.HexColor("#FFFFFF")))

    table.setStyle(TableStyle(ts))
    elements.append(table)

    # 4. Footer Note
    elements.append(Spacer(1, 16))
    footer_style = ParagraphStyle(
        "FooterNote",
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#94A3B8"),
    )
    elements.append(
        Paragraph(
            "This report was automatically generated via Document Forensics AI (Gemini 1.5 Flash Vision & ReportLab). "
            "Verified for submission to KX management system.",
            footer_style,
        )
    )

    doc.build(elements)
    buffer.seek(0)
    return buffer
