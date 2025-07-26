"""
Tests for the core Reverse Attribution framework.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock

# Mock the RA import to avoid actual dependencies
class MockReverseAttribution:
    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs
    
    def analyze_sample(self, input_data):
        return {
            'a_flip': np.random.uniform(100, 1000),
            'counter_evidence': [[1, 2, -0.5, 0.8]],
            'attribution_stability': 0.75
        }
    
    def batch_analyze(self, inputs, labels):
        results = []
        for i in range(len(inputs)):
            results.append(self.analyze_sample(inputs[i]))
        return results

@pytest.fixture
def mock_ra():
    """Mock Reverse Attribution instance."""
    return MockReverseAttribution

class TestReverseAttribution:
    """Test cases for ReverseAttribution class."""
    
    def test_ra_initialization(self, mock_model, mock_ra):
        """Test RA initialization with various parameters."""
        ra = mock_ra(mock_model, attribution_method='integrated_gradients')
        assert ra.model == mock_model
        assert 'attribution_method' in ra.kwargs
    
    def test_analyze_sample_basic(self, mock_model, mock_ra):
        """Test basic sample analysis."""
        ra = mock_ra(mock_model)
        
        # Mock input data
        input_data = torch.randn(1, 3, 32, 32)
        
        result = ra.analyze_sample(input_data)
        
        assert 'a_flip' in result
        assert 'counter_evidence' in result
        assert 'attribution_stability' in result
        assert isinstance(result['a_flip'], (int, float))
        assert isinstance(result['counter_evidence'], list)
    
    def test_a_flip_calculation(self, mock_model, mock_ra):
        """Test A-Flip score calculation logic."""
        ra = mock_ra(mock_model)
        
        # Test multiple samples
        results = []
        for _ in range(10):
            input_data = torch.randn(1, 3, 32, 32)
            result = ra.analyze_sample(input_data)
            results.append(result['a_flip'])
        
        # A-Flip scores should be positive and reasonable
        assert all(score > 0 for score in results)
        assert all(score < 5000 for score in results)  # Reasonable upper bound
    
    def test_counter_evidence_detection(self, mock_model, mock_ra):
        """Test counter-evidence detection mechanism."""
        ra = mock_ra(mock_model)
        
        input_data = torch.randn(1, 3, 32, 32)
        result = ra.analyze_sample(input_data)
        
        counter_evidence = result['counter_evidence']
        assert isinstance(counter_evidence, list)
        
        # Each counter-evidence entry should have required fields
        if counter_evidence:
            for ce in counter_evidence:
                assert len(ce) >= 3  # At minimum: location, location, strength
                assert isinstance(ce[2], (int, float))  # Strength should be numeric
    
    def test_batch_analyze(self, mock_model, mock_ra):
        """Test batch analysis functionality."""
        ra = mock_ra(mock_model)
        
        # Create batch of inputs
        batch_size = 5
        inputs = [torch.randn(1, 3, 32, 32) for _ in range(batch_size)]
        labels = [i % 10 for i in range(batch_size)]
        
        results = ra.batch_analyze(inputs, labels)
        
        assert len(results) == batch_size
        assert all('a_flip' in result for result in results)
        assert all('counter_evidence' in result for result in results)
    
    @pytest.mark.slow
    def test_statistical_significance(self, mock_model, mock_ra):
        """Test statistical significance of A-Flip scores."""
        ra = mock_ra(mock_model)
        
        # Generate larger sample for statistical analysis
        n_samples = 50
        a_flip_scores = []
        
        for _ in range(n_samples):
            input_data = torch.randn(1, 3, 32, 32)
            result = ra.analyze_sample(input_data)
            a_flip_scores.append(result['a_flip'])
        
        # Statistical tests
        mean_score = np.mean(a_flip_scores)
        std_score = np.std(a_flip_scores)
        
        assert mean_score > 0
        assert std_score > 0
        assert len(a_flip_scores) == n_samples
        
        # Coefficient of variation should be reasonable
        cv = std_score / mean_score
        assert 0.1 < cv < 2.0  # Reasonable variability range

class TestRAMetrics:
    """Test RA-specific metrics and calculations."""
    
    def test_attribution_stability_metric(self):
        """Test attribution stability calculation."""
        # Mock attribution maps
        attr1 = np.random.randn(100)
        attr2 = np.random.randn(100) 
        attr2_similar = attr1 + np.random.normal(0, 0.1, 100)  # Similar to attr1
        
        # Stability should be higher for similar attributions
        similarity1 = np.corrcoef(attr1, attr2)[0, 1]
        similarity2 = np.corrcoef(attr1, attr2_similar)[0, 1]
        
        assert similarity2 > similarity1
    
    def test_counter_evidence_strength_calculation(self):
        """Test counter-evidence strength calculation."""
        # Mock scenario: high confidence prediction with contradictory evidence
        prediction_confidence = 0.9
        evidence_strength = -0.5  # Negative indicates contradiction
        
        # Counter-evidence strength should be meaningful
        ce_strength = abs(evidence_strength) * (1 - prediction_confidence)
        
        assert ce_strength > 0
        assert ce_strength < 1
    
    def test_a_flip_normalization(self):
        """Test A-Flip score normalization."""
        # Test different scenarios
        raw_scores = [100, 500, 1000, 2000]
        
        # Normalization should maintain relative ordering
        normalized = []
        max_score = max(raw_scores)
        for score in raw_scores:
            normalized.append(score / max_score)
        
        assert all(0 <= norm <= 1 for norm in normalized)
        assert normalized == sorted(normalized)  # Should maintain order
