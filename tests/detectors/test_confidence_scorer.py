"""
Unit tests for ConfidenceScorer
"""
import pytest


class TestConfidenceScorer:
    """Test suite for ConfidenceScorer class."""
    
    def test_initialization(self, confidence_scorer):
        """Test that ConfidenceScorer initializes correctly."""
        assert confidence_scorer is not None
        assert hasattr(confidence_scorer, 'STRONG_INDICATORS')
        assert hasattr(confidence_scorer, 'WEAK_INDICATORS')
        assert hasattr(confidence_scorer, 'INDICATOR_WEIGHTS')
    
    def test_calculate_confidence_empty_results(self, confidence_scorer):
        """Test confidence calculation with empty results."""
        empty_results = {}
        result = confidence_scorer.calculate_confidence(empty_results)
        
        assert result is not None
        assert 'confidence_score' in result
        assert 'confidence_level' in result
        assert 'recommendation' in result
    
    def test_calculate_confidence_basic(self, confidence_scorer):
        """Test basic confidence calculation."""
        results = {
            'metadata': {
                'flags': ['suspicious_software'],
                'risk_score': 50
            },
            'pixel': {
                'flags': [],
                'variance': 10.5
            }
        }
        
        result = confidence_scorer.calculate_confidence(results)
        
        assert result is not None
        assert 'confidence_score' in result
        assert 'confidence_level' in result
        assert 'indicator_count' in result
        assert isinstance(result['confidence_score'], (int, float))
        assert 0 <= result['confidence_score'] <= 100
    
    def test_calculate_confidence_high_suspicion(self, confidence_scorer):
        """Test confidence calculation with high suspicion indicators."""
        results = {
            'metadata': {
                'flags': ['suspicious_software', 'timeline_anomaly'],
                'risk_score': 80
            },
            'structure': {
                'flags': ['xref_anomalies'],
                'risk_score': 70
            },
            'pixel': {
                'flags': ['smoothing_detected'],
                'variance': 2.0
            }
        }
        
        result = confidence_scorer.calculate_confidence(results)
        
        assert result['confidence_score'] > 50  # Should be high
        assert result['confidence_level'] in ['High Suspicion', 'Definitive Fraud', 'Moderate Suspicion']
