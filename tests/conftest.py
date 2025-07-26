"""
Pytest configuration and shared fixtures for Reverse Attribution tests.
"""

import pytest
import json
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock
import torch

# Test data fixtures
@pytest.fixture
def sample_evaluation_results():
    """Sample evaluation results for testing."""
    return {
        "cifar10_results": {
            "standard_metrics": {
                "accuracy": 0.9328,
                "precision": 0.9301,
                "recall": 0.9328,
                "f1": 0.9314,
                "ece": 0.0451,
                "brier_score": 0.0112,
                "model_type": "ResNet",
                "architecture": "ResNet-56",
                "total_parameters": 855770,
                "num_samples": 10000
            },
            "ra_analysis": {
                "summary": {
                    "avg_a_flip": 819.42,
                    "std_a_flip": 300.34,
                    "avg_counter_evidence_count": 5.0,
                    "avg_counter_evidence_strength": -0.15,
                    "samples_analyzed": 200
                },
                "detailed_results": [
                    {
                        "sample_id": 0,
                        "y_hat": 1,
                        "runner_up": 9,
                        "a_flip": 241.12,
                        "counter_evidence": [
                            [15, 20, -2.1, 0.85],
                            [22, 18, -1.8, 0.72]
                        ]
                    }
                ]
            }
        },
        "imdb_results": {
            "standard_metrics": {
                "accuracy": 0.9156,
                "precision": 0.0000,
                "recall": 0.0000,
                "f1": 0.0000,
                "model_type": "BERT",
                "architecture": "BERT-base-uncased",
                "total_parameters": 110000000
            },
            "ra_analysis": {
                "summary": {
                    "avg_a_flip": 0.0,
                    "std_a_flip": 0.0,
                    "samples_analyzed": 0
                }
            }
        }
    }

@pytest.fixture
def sample_jmlr_metrics():
    """Sample JMLR metrics for testing."""
    return {
        "cifar10": {
            "accuracy": 0.9328,
            "ece": 0.0451,
            "brier_score": 0.0112,
            "avg_a_flip": 819.42,
            "std_a_flip": 300.34,
            "avg_counter_evidence_count": 5.0,
            "samples_analyzed": 200,
            "model_type": "ResNet-56"
        }
    }

@pytest.fixture
def temp_results_dir():
    """Create temporary directory with sample result files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create sample files
        eval_results = {
            "cifar10_results": {
                "standard_metrics": {"accuracy": 0.93, "f1": 0.92}
            }
        }
        
        with open(temp_path / "evaluation_results.json", 'w') as f:
            json.dump(eval_results, f)
            
        with open(temp_path / "jmlr_metrics.json", 'w') as f:
            json.dump({"cifar10": {"accuracy": 0.93}}, f)
            
        yield temp_path

@pytest.fixture
def mock_model():
    """Mock model for testing."""
    model = Mock()
    model.parameters.return_value = [torch.randn(100, 50), torch.randn(50)]
    model.eval.return_value = model
    return model

@pytest.fixture
def mock_visualizer_data():
    """Mock data for visualizer testing."""
    return {
        'models': {
            'cifar10': {
                'config': {
                    'name': 'CIFAR-10 ResNet',
                    'color': '#F18F01',
                    'architecture': 'ResNet-56'
                },
                'performance_metrics': {
                    'accuracy': 0.93,
                    'f1': 0.92,
                    'precision': 0.94,
                    'recall': 0.91
                },
                'ra_metrics': {
                    'avg_a_flip': 819.42,
                    'samples_analyzed': 200
                }
            }
        }
    }

# Custom pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
