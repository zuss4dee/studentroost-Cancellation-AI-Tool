"""
Integration tests for the main application
"""
import pytest
import sys
from pathlib import Path
from io import BytesIO

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestAppFunctions:
    """Test suite for app.py utility functions."""
    
    def test_get_file_type_pdf(self):
        """Test file type detection for PDF."""
        from app import get_file_type
        assert get_file_type('document.pdf') == 'pdf'
        assert get_file_type('test.PDF') == 'pdf'
    
    def test_get_file_type_image(self):
        """Test file type detection for images."""
        from app import get_file_type
        assert get_file_type('image.jpg') == 'image'
        assert get_file_type('photo.JPEG') == 'image'
        assert get_file_type('picture.png') == 'image'
        assert get_file_type('scan.tiff') == 'image'
    
    def test_get_file_type_unknown(self):
        """Test file type detection for unknown formats."""
        from app import get_file_type
        assert get_file_type('file.txt') == 'unknown'
        assert get_file_type('document') == 'unknown'
    
    def test_pdf_to_image(self, sample_pdf_bytes):
        """Test PDF to image conversion."""
        from app import pdf_to_image
        pdf_bytes = sample_pdf_bytes.read()
        img = pdf_to_image(pdf_bytes)
        
        assert img is not None
        from PIL import Image
        assert isinstance(img, Image.Image)
    
    def test_analyze_file_pdf(self, sample_pdf_bytes):
        """Test full file analysis for PDF."""
        from app import analyze_file
        
        # Create a mock file-like object
        class MockFile:
            def __init__(self, bytes_data, name):
                self.bytes_data = bytes_data
                self.name = name
                self._position = 0
            
            def read(self):
                return self.bytes_data
        
        mock_file = MockFile(sample_pdf_bytes.read(), 'test.pdf')
        analysis = analyze_file(mock_file)
        
        assert 'filename' in analysis
        assert 'file_type' in analysis
        assert 'metadata' in analysis
        assert 'noise' in analysis
        assert analysis['file_type'] == 'pdf'
