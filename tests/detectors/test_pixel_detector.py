"""
Unit tests for PixelDetector
"""
import pytest
from PIL import Image
import numpy as np


class TestPixelDetector:
    """Test suite for PixelDetector class."""
    
    def test_initialization(self, pixel_detector):
        """Test that PixelDetector initializes correctly."""
        assert pixel_detector is not None
    
    def test_analyze_ela(self, pixel_detector, sample_image):
        """Test Error Level Analysis."""
        ela_result = pixel_detector.analyze_ela(sample_image)
        
        assert ela_result is not None
        assert isinstance(ela_result, Image.Image)
        assert ela_result.mode == 'RGB'
    
    def test_analyze_ela_different_modes(self, pixel_detector):
        """Test ELA with different image modes."""
        # Test with RGBA
        rgba_img = Image.new('RGBA', (50, 50), (255, 0, 0, 128))
        ela_result = pixel_detector.analyze_ela(rgba_img)
        assert ela_result.mode == 'RGB'
        
        # Test with grayscale
        gray_img = Image.new('L', (50, 50), 128)
        ela_result = pixel_detector.analyze_ela(gray_img)
        assert ela_result.mode == 'RGB'
    
    def test_analyze_noise(self, pixel_detector, sample_image):
        """Test noise variance analysis."""
        result = pixel_detector.analyze_noise(sample_image)
        
        assert 'variance' in result
        assert 'flags' in result
        assert 'findings' in result
        assert isinstance(result['variance'], (int, float))
        assert isinstance(result['flags'], list)
        assert isinstance(result['findings'], str)
        assert result['variance'] >= 0
    
    def test_analyze_noise_smooth_image(self, pixel_detector):
        """Test noise analysis on a smooth (potentially edited) image."""
        # Create a very smooth image (low noise)
        smooth_array = np.ones((100, 100, 3), dtype=np.uint8) * 128
        smooth_img = Image.fromarray(smooth_array)
        
        result = pixel_detector.analyze_noise(smooth_img)
        
        assert result['variance'] >= 0
        # Smooth images should have low variance
        assert result['variance'] < 100  # Very low for uniform image
    
    def test_analyze_noise_noisy_image(self, pixel_detector):
        """Test noise analysis on a noisy (natural) image."""
        # Create a noisy image
        noisy_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        noisy_img = Image.fromarray(noisy_array)
        
        result = pixel_detector.analyze_noise(noisy_img)
        
        assert result['variance'] >= 0
        # Noisy images should have higher variance
        assert result['variance'] > 0
