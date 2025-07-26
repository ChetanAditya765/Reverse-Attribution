"""
Tests for model-related functionality including parameter counting and architecture validation.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

class MockBERTModel(nn.Module):
    """Mock BERT model for testing."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.bert = Mock()
        self.classifier = nn.Linear(768, num_classes)
        self.num_classes = num_classes
    
    def forward(self, input_ids, attention_mask=None):
        # Mock BERT output
        batch_size = input_ids.size(0)
        hidden_states = torch.randn(batch_size, 768)
        return self.classifier(hidden_states)

class MockResNetModel(nn.Module):
    """Mock ResNet model for testing."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64, num_classes)
        self.num_classes = num_classes
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class TestModelParameterCounting:
    """Test model parameter counting functionality."""
    
    def test_bert_parameter_counting(self):
        """Test parameter counting for BERT models."""
        model = MockBERTModel(num_classes=2)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params == total_params  # All should be trainable by default
    
    def test_resnet_parameter_counting(self):
        """Test parameter counting for ResNet models."""
        model = MockResNetModel(num_classes=10)
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # Expected parameter count should be reasonable for our mock model
        assert total_params > 1000  # Should have reasonable number of parameters
        assert total_params < 1000000  # But not too many for a mock model
    
    def test_parameter_count_by_layer(self):
        """Test parameter counting by layer."""
        model = MockResNetModel()
        
        layer_params = {}
        for name, param in model.named_parameters():
            layer_params[name] = param.numel()
        
        assert len(layer_params) > 0
        assert all(count > 0 for count in layer_params.values())
    
    def test_model_size_estimation(self):
        """Test model size estimation in MB."""
        model = MockResNetModel()
        
        total_params = sum(p.numel() for p in model.parameters())
        # Assume float32 (4 bytes per parameter)
        model_size_mb = (total_params * 4) / (1024 * 1024)
        
        assert model_size_mb > 0
        assert model_size_mb < 1000  # Should be reasonable size

class TestModelArchitectureValidation:
    """Test model architecture validation."""
    
    def test_bert_architecture_validation(self):
        """Test BERT model architecture validation."""
        model = MockBERTModel(num_classes=2)
        
        # Check model structure
        assert hasattr(model, 'bert')
        assert hasattr(model, 'classifier')
        assert hasattr(model, 'num_classes')
        assert model.num_classes == 2
    
    def test_resnet_architecture_validation(self):
        """Test ResNet model architecture validation."""
        model = MockResNetModel(num_classes=10)
        
        # Check model structure
        assert hasattr(model, 'features')
        assert hasattr(model, 'classifier')
        assert hasattr(model, 'num_classes')
        assert model.num_classes == 10
    
    def test_model_forward_pass(self):
        """Test model forward pass functionality."""
        # Test ResNet forward pass
        resnet_model = MockResNetModel(num_classes=10)
        input_tensor = torch.randn(1, 3, 32, 32)
        output = resnet_model(input_tensor)
        
        assert output.shape == (1, 10)
        
        # Test BERT forward pass
        bert_model = MockBERTModel(num_classes=2)
        input_ids = torch.randint(0, 1000, (1, 128))
        attention_mask = torch.ones(1, 128)
        output = bert_model(input_ids, attention_mask)
        
        assert output.shape == (1, 2)
    
    def test_model_device_compatibility(self):
        """Test model device compatibility."""
        model = MockResNetModel()
        
        # Test CPU
        assert next(model.parameters()).device == torch.device('cpu')
        
        # Test GPU if available
        if torch.cuda.is_available():
            model_gpu = model.cuda()
            assert next(model_gpu.parameters()).device.type == 'cuda'

class TestModelConfiguration:
    """Test model configuration and setup."""
    
    def test_model_config_validation(self):
        """Test model configuration validation."""
        configs = [
            {
                'name': 'IMDb BERT',
                'architecture': 'BERT-base-uncased',
                'num_classes': 2,
                'expected_parameters': 110000000
            },
            {
                'name': 'CIFAR-10 ResNet',
                'architecture': 'ResNet-56',
                'num_classes': 10,
                'expected_parameters': 855770
            }
        ]
        
        for config in configs:
            assert 'name' in config
            assert 'architecture' in config
            assert 'num_classes' in config
            assert 'expected_parameters' in config
            assert config['num_classes'] > 0
            assert config['expected_parameters'] > 0
    
    def test_model_initialization_parameters(self):
        """Test model initialization with different parameters."""
        # Test different num_classes
        for num_classes in [2, 5, 10]:
            bert_model = MockBERTModel(num_classes=num_classes)
            resnet_model = MockResNetModel(num_classes=num_classes)
            
            assert bert_model.num_classes == num_classes
            assert resnet_model.num_classes == num_classes
            assert bert_model.classifier.out_features == num_classes
            assert resnet_model.classifier.out_features == num_classes

class TestModelEvaluation:
    """Test model evaluation functionality."""
    
    def test_model_eval_mode(self):
        """Test setting model to evaluation mode."""
        model = MockResNetModel()
        
        # Set to eval mode
        model.eval()
        
        # Check training flag
        assert not model.training
        
        # Set back to train mode
        model.train()
        assert model.training
    
    def test_model_inference(self):
        """Test model inference with no gradients."""
        model = MockResNetModel()
        model.eval()
        
        input_tensor = torch.randn(1, 3, 32, 32)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape == (1, 10)
        assert not output.requires_grad
    
    def test_model_batch_processing(self):
        """Test model batch processing capabilities."""
        model = MockResNetModel()
        
        # Test different batch sizes
        for batch_size in [1, 4, 8]:
            input_tensor = torch.randn(batch_size, 3, 32, 32)
            output = model(input_tensor)
            
            assert output.shape == (batch_size, 10)
