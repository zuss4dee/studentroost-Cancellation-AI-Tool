"""
Pytest configuration and shared fixtures
"""
import pytest
import sys
from pathlib import Path
from io import BytesIO
from PIL import Image
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_image():
    """Create a sample PIL Image for testing."""
    # Create a simple 100x100 RGB image
    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


@pytest.fixture
def sample_image_bytes(sample_image):
    """Convert sample image to bytes."""
    buffer = BytesIO()
    sample_image.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer


@pytest.fixture
def sample_pdf_bytes():
    """Create a minimal PDF in memory for testing."""
    try:
        import fitz
        # Create a simple PDF document
        doc = fitz.open()
        page = doc.new_page(width=200, height=200)
        page.insert_text((50, 50), "Test PDF Document")
        pdf_bytes = doc.tobytes()
        doc.close()
        return BytesIO(pdf_bytes)
    except ImportError:
        pytest.skip("PyMuPDF not available")


@pytest.fixture
def metadata_detector():
    """Create a MetadataDetector instance."""
    from detectors.metadata_detector import MetadataDetector
    return MetadataDetector()


@pytest.fixture
def pixel_detector():
    """Create a PixelDetector instance."""
    from detectors.pixel_detector import PixelDetector
    return PixelDetector()


@pytest.fixture
def confidence_scorer():
    """Create a ConfidenceScorer instance."""
    from detectors.confidence_scorer import ConfidenceScorer
    return ConfidenceScorer()
