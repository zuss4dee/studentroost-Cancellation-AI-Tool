"""
Unit tests for MetadataDetector
"""
import pytest
from io import BytesIO
from PIL import Image
import numpy as np


class TestMetadataDetector:
    """Test suite for MetadataDetector class."""
    
    def test_initialization(self, metadata_detector):
        """Test that MetadataDetector initializes correctly."""
        assert metadata_detector is not None
        assert hasattr(metadata_detector, 'SUSPICIOUS_SOFTWARE')
        assert hasattr(metadata_detector, 'TRUSTED_SOFTWARE')
        assert hasattr(metadata_detector, 'INSTITUTIONAL_PATTERNS')
    
    def test_analyze_unsupported_format(self, metadata_detector):
        """Test analysis of unsupported file format."""
        file_stream = BytesIO(b"invalid file content")
        result = metadata_detector.analyze(file_stream, 'unknown', 'test.xyz')
        
        assert result['risk_score'] == 0
        assert 'Unsupported File Format' in result['flags']
        assert result['trust_score'] == 0
    
    def test_analyze_image_basic(self, metadata_detector, sample_image_bytes):
        """Test basic image metadata analysis."""
        result = metadata_detector.analyze(sample_image_bytes, 'image', 'test.png')
        
        assert 'risk_score' in result
        assert 'flags' in result
        assert 'raw_data' in result
        assert 'trust_score' in result
        assert isinstance(result['risk_score'], (int, float))
        assert isinstance(result['flags'], list)
        assert isinstance(result['raw_data'], dict)
        assert 0 <= result['risk_score'] <= 100
        assert 0 <= result['trust_score'] <= 100
    
    def test_analyze_pdf_basic(self, metadata_detector, sample_pdf_bytes):
        """Test basic PDF metadata analysis."""
        result = metadata_detector.analyze(sample_pdf_bytes, 'pdf', 'test.pdf')
        
        assert 'risk_score' in result
        assert 'flags' in result
        assert 'raw_data' in result
        assert 'trust_score' in result
        assert isinstance(result['risk_score'], (int, float))
        assert isinstance(result['flags'], list)
        assert isinstance(result['raw_data'], dict)
    
    def test_suspicious_software_detection(self, metadata_detector):
        """Test detection of suspicious software in metadata."""
        # This would require creating a PDF/image with specific metadata
        # For now, we test that the detector has the capability
        assert 'photoshop' in metadata_detector.SUSPICIOUS_SOFTWARE
        assert 'gimp' in metadata_detector.SUSPICIOUS_SOFTWARE
    
    def test_institutional_patterns(self, metadata_detector):
        """Test institutional pattern detection."""
        assert 'university' in metadata_detector.INSTITUTIONAL_PATTERNS
        assert 'home office' in metadata_detector.INSTITUTIONAL_PATTERNS
    
    def test_find_institutional_indicators(self, metadata_detector):
        """Test finding institutional indicators in text."""
        text = "This is a document from University of London"
        filename = "cas_document.pdf"
        
        indicators = metadata_detector._find_institutional_indicators(text, filename)
        
        assert isinstance(indicators, list)
        # Should find 'university' and potentially 'cas' pattern
        assert len(indicators) > 0
    
    def test_error_handling(self, metadata_detector):
        """Test error handling with invalid input."""
        # Test with None
        invalid_stream = BytesIO(b"")
        result = metadata_detector.analyze(invalid_stream, 'image', 'test.png')
        
        # Should return a result structure even on error
        assert 'risk_score' in result
        assert 'flags' in result
