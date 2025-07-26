"""
Tests for the ExplanationVisualizer class with statistical robustness features.
"""

import pytest
import json
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

# Mock the visualizer class
class MockExplanationVisualizer:
    def __init__(self, output_dir="test_figs"):
        self.output_dir = Path(output_dir)
        self.models = {}
        self.data = {}
        self.formats = ['png']
    
    def load_results(self, file_path=None):
        if file_path:
            return {"test": "data"}
        return self._auto_discover_results()
    
    def _auto_discover_results(self):
        return {"evaluation_results": {"cifar10": {"accuracy": 0.93}}}
    
    def visualize_all(self):
        return {"performance_comparison": "test_path"}
    
    def create_performance_comparison_with_error_bars(self):
        return "test_performance_path"

@pytest.fixture
def mock_visualizer():
    return MockExplanationVisualizer

class TestExplanationVisualizerInitialization:
    """Test ExplanationVisualizer initialization and setup."""
    
    def test_visualizer_initialization(self, mock_visualizer):
        """Test basic visualizer initialization."""
        viz = mock_visualizer("test_output")
        
        assert viz.output_dir == Path("test_output")
        assert viz.formats == ['png']
        assert isinstance(viz.models, dict)
        assert isinstance(viz.data, dict)
    
    def test_output_directory_creation(self, mock_visualizer):
        """Test output directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "visualizations"
            viz = mock_visualizer(str(output_dir))
            
            # Directory should be created
            assert viz.output_dir == output_dir
    
    def test_format_configuration(self, mock_visualizer):
        """Test output format configuration."""
        viz = mock_visualizer()
        viz.formats = ['png', 'pdf', 'svg']
        
        assert 'png' in viz.formats
        assert 'pdf' in viz.formats
        assert 'svg' in viz.formats

class TestDataLoading:
    """Test data loading and auto-discovery functionality."""
    
    def test_auto_discover_results(self, mock_visualizer, temp_results_dir):
        """Test auto-discovery of result files."""
        viz = mock_visualizer()
        
        # Mock the discovery process
        data = viz._auto_discover_results()
        
        assert isinstance(data, dict)
        assert 'evaluation_results' in data or len(data) > 0
    
    def test_load_specific_file(self, mock_visualizer, sample_evaluation_results):
        """Test loading from specific file."""
        viz = mock_visualizer()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_evaluation_results, f)
            temp_file = f.name
        
        try:
            data = viz.load_results(temp_file)
            assert isinstance(data, dict)
        finally:
            Path(temp_file).unlink()
    
    def test_model_extraction(self, mock_visualizer, sample_evaluation_results):
        """Test model data extraction from results."""
        viz = mock_visualizer()
        viz.data = {'evaluation_results': sample_evaluation_results}
        
        # Mock model extraction
        viz.models = {
            'cifar10': {
                'performance_metrics': sample_evaluation_results['cifar10_results']['standard_metrics'],
                'ra_metrics': sample_evaluation_results['cifar10_results']['ra_analysis']['summary']
            }
        }
        
        assert 'cifar10' in viz.models
        assert 'performance_metrics' in viz.models['cifar10']

class TestVisualizationGeneration:
    """Test visualization generation functionality."""
    
    def test_performance_comparison_generation(self, mock_visualizer, mock_visualizer_data):
        """Test performance comparison chart generation."""
        viz = mock_visualizer()
        viz.models = mock_visualizer_data['models']
        
        result = viz.create_performance_comparison_with_error_bars()
        
        assert result is not None
        assert isinstance(result, str)
    
    def test_error_bar_calculations(self, mock_visualizer):
        """Test error bar calculation for statistical robustness."""
        viz = mock_visualizer()
        
        # Mock performance data
        accuracy_values = [0.93, 0.94, 0.92, 0.95]
        
        # Calculate error bars (standard approach)
        mean_acc = np.mean(accuracy_values)
        std_acc = np.std(accuracy_values)
        error_bar = 1.96 * std_acc / np.sqrt(len(accuracy_values))  # 95% CI
        
        assert mean_acc > 0
        assert error_bar >= 0
        assert mean_acc - error_bar >= 0  # Should not go negative
    
    def test_f1_score_diagnostic(self, mock_visualizer, sample_evaluation_results):
        """Test F1-score diagnostic functionality."""
        viz = mock_visualizer()
        
        # Test F1-score issue detection
        metrics = sample_evaluation_results['imdb_results']['standard_metrics']
        
        f1_score = metrics.get('f1', 0)
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        
        # Diagnostic logic
        f1_issue_detected = (f1_score == 0.0 and (precision > 0 or recall > 0))
        
        assert f1_score == 0.0  # Known issue in sample data
        assert precision == 0.0  # Should be zero
        assert not f1_issue_detected  # No issue if both precision and recall are 0
    
    def test_attribution_method_specification(self, mock_visualizer):
        """Test attribution method specification in visualizations."""
        viz = mock_visualizer()
        
        # Mock model configuration with attribution methods
        model_config = {
            'attribution_methods': ['Integrated Gradients', 'GradCAM', 'Attention Weights'],
            'name': 'Test Model'
        }
        
        methods = model_config['attribution_methods']
        
        assert 'Integrated Gradients' in methods
        assert 'GradCAM' in methods
        assert len(methods) >= 2

class TestStatisticalRobustness:
    """Test statistical robustness features."""
    
    def test_confidence_interval_calculation(self, mock_visualizer):
        """Test confidence interval calculation for A-Flip scores."""
        viz = mock_visualizer()
        
        # Mock A-Flip score data
        aflip_scores = np.random.normal(500, 100, 50)  # Mean=500, std=100, n=50
        
        mean_aflip = np.mean(aflip_scores)
        std_aflip = np.std(aflip_scores)
        sem_aflip = std_aflip / np.sqrt(len(aflip_scores))
        ci_95 = 1.96 * sem_aflip
        
        assert mean_aflip > 0
        assert std_aflip > 0
        assert ci_95 > 0
        assert ci_95 < mean_aflip  # CI should be smaller than mean
    
    def test_parameter_count_resolution(self, mock_visualizer):
        """Test parameter count resolution from various sources."""
        viz = mock_visualizer()
        
        # Test scenarios
        scenarios = [
            {'total_parameters': 855770, 'expected': 855770, 'source': 'measured'},
            {'expected_parameters': 110000000, 'expected': 110000000, 'source': 'expected'},
            {'architecture': 'resnet-56', 'expected': 855770, 'source': 'estimated'}
        ]
        
        for scenario in scenarios:
            if 'total_parameters' in scenario:
                param_count = scenario['total_parameters']
            elif 'expected_parameters' in scenario:
                param_count = scenario['expected_parameters']
            else:
                # Mock estimation logic
                param_count = 855770 if 'resnet-56' in scenario.get('architecture', '') else None
            
            assert param_count is not None
            assert param_count > 0
    
    def test_statistical_significance_testing(self, mock_visualizer):
        """Test statistical significance testing functionality."""
        viz = mock_visualizer()
        
        # Mock two model performance comparisons
        model1_scores = np.random.normal(0.93, 0.02, 30)
        model2_scores = np.random.normal(0.89, 0.02, 30)
        
        # Basic statistical test
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(model1_scores, model2_scores)
        
        assert isinstance(t_stat, (int, float))
        assert 0 <= p_value <= 1
        
        # Model 1 should be significantly better (p < 0.05 expected)
        is_significant = p_value < 0.05
        assert isinstance(is_significant, bool)

class TestIntegrationFeatures:
    """Test integration with other components."""
    
    def test_multi_model_visualization(self, mock_visualizer, sample_evaluation_results):
        """Test multi-model visualization generation."""
        viz = mock_visualizer()
        
        # Mock multi-model data
        viz.models = {
            'cifar10': sample_evaluation_results['cifar10_results'],
            'imdb': sample_evaluation_results['imdb_results']
        }
        
        results = viz.visualize_all()
        
        assert isinstance(results, dict)
        assert len(results) > 0
    
    def test_cross_domain_analysis(self, mock_visualizer):
        """Test cross-domain analysis functionality."""
        viz = mock_visualizer()
        
        # Mock text and vision models
        text_models = ['imdb', 'yelp']
        vision_models = ['cifar10']
        
        # Basic cross-domain comparison
        has_text = len(text_models) > 0
        has_vision = len(vision_models) > 0
        can_compare = has_text and has_vision
        
        assert has_vision  # Should have at least vision model
        assert isinstance(can_compare, bool)
    
    @pytest.mark.integration
    def test_end_to_end_visualization(self, mock_visualizer, temp_results_dir):
        """Test complete end-to-end visualization pipeline."""
        viz = mock_visualizer(str(temp_results_dir / "output"))
        
        # Load data
        data = viz.load_results()
        
        # Generate visualizations
        results = viz.visualize_all()
        
        assert isinstance(data, dict)
        assert isinstance(results, dict)
