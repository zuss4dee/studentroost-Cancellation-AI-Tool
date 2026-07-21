from io import BytesIO
from backend.src.report_generator import generate_translated_id_pdf
from backend.src.foreign_id_translator import clean_json_response


def test_clean_json_response():
    raw_markdown = """```json
    {
      "Document Type": "Passport",
      "First Name": "Jean",
      "Last Name": "Dupont",
      "Date of Birth": "1990-05-15"
    }
    ```"""
    cleaned = clean_json_response(raw_markdown)
    assert cleaned.startswith("{")
    assert cleaned.endswith("}")
    assert '"Passport"' in cleaned


def test_generate_translated_id_pdf():
    sample_data = {
        "Document Type": "National Identity Card",
        "First Name": "Maria",
        "Last Name": "Garcia",
        "Date of Birth": "1988-11-20",
        "Nationality": "Spanish",
        "Document Number": "ID-987654321",
        "Expiry Date": "2030-11-20",
    }
    pdf_buffer = generate_translated_id_pdf(sample_data, source_filename="spanish_id.jpg")
    assert isinstance(pdf_buffer, BytesIO)
    pdf_bytes = pdf_buffer.getvalue()
    assert len(pdf_bytes) > 0
    # Check PDF header bytes %PDF-
    assert pdf_bytes.startswith(b"%PDF-")


def test_render_overlay_image():
    from PIL import Image
    from backend.src.translated_id_overlay import render_overlay_image

    img = Image.new("RGB", (800, 600), color=(240, 240, 240))
    sample_boxes = [
        {
            "box_2d": [100, 200, 200, 600],
            "original_text": "APELLIDO",
            "translated_text": "Surname: GARCIA",
        }
    ]
    annotated, count_drawn, placements = render_overlay_image(img, sample_boxes)
    assert annotated.size == (800, 600)  # Output canvas is original ID image
    assert count_drawn == 1
    assert len(placements) == 1





