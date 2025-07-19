"""
Model Factory that properly integrates with your actual model implementations.
Now correctly imports and uses your BERTSentimentClassifier and ResNetCIFAR models.
"""

import torch
import torch.nn as nn
import os
from typing import Optional, Union, Dict, Any
from pathlib import Path
import sys

# Add models directory to path to import your actual models
models_path = Path(__file__).parent.parent / "models"
sys.path.insert(0, str(models_path))

# Import your actual model implementations
try:
    from models.bert_sentiment import BERTSentimentClassifier, create_bert_sentiment_model
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("⚠️ BERT sentiment model not found in models/")

try:
    from models.resnet_cifar import ResNetCIFAR, resnet56_cifar, resnet20_cifar, resnet32_cifar
    RESNET_AVAILABLE = True
except ImportError:
    RESNET_AVAILABLE = False
    print("⚠️ ResNet CIFAR models not found in models/")

try:
    from models.custom_model_example import CustomTextClassifier, CustomVisionClassifier
    CUSTOM_AVAILABLE = True
except ImportError:
    CUSTOM_AVAILABLE = False
    print("⚠️ Custom model examples not found in models/")


class ModelFactory:
    """
    Factory class that creates instances of your actual model implementations.
    """
    
    @staticmethod
    def create_text_model(
        model_name: str = "bert-base-uncased",
        num_classes: int = 2,
        checkpoint_path: Optional[str] = None,
        model_type: str = "bert_sentiment",
        **kwargs
    ) -> nn.Module:
        """
        Create text classification model using your actual implementations.
        
        Args:
            model_name: HuggingFace model name or custom model identifier
            num_classes: Number of output classes
            checkpoint_path: Path to saved model weights
            model_type: Type of model ("bert_sentiment", "custom_text")
            **kwargs: Additional model arguments
            
        Returns:
            Your actual model instance
        """
        
        if model_type == "bert_sentiment" and BERT_AVAILABLE:
            # Use your BERTSentimentClassifier
            model = BERTSentimentClassifier(
                model_name=model_name,
                num_classes=num_classes,
                **kwargs
            )
            
            # Load checkpoint if provided
            if checkpoint_path and os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            
            return model
            
        elif model_type == "custom_text" and CUSTOM_AVAILABLE:
            # Use your CustomTextClassifier
            model = CustomTextClassifier(
                num_classes=num_classes,
                **kwargs
            )
            
            if checkpoint_path and os.path.exists(checkpoint_path):
                model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
            
            return model
            
        else:
            raise ValueError(f"Model type '{model_type}' not available or not implemented")
    
    @staticmethod
    def create_vision_model(
        architecture: str = "resnet56",
        num_classes: int = 10,
        checkpoint_path: Optional[str] = None,
        model_type: str = "resnet_cifar",
        **kwargs
    ) -> nn.Module:
        """
        Create vision classification model using your actual implementations.
        
        Args:
            architecture: Model architecture ("resnet56", "resnet20", etc.)
            num_classes: Number of output classes
            checkpoint_path: Path to saved model weights
            model_type: Type of model ("resnet_cifar", "custom_vision")
            **kwargs: Additional model arguments
            
        Returns:
            Your actual model instance
        """
        
        if model_type == "resnet_cifar" and RESNET_AVAILABLE:
            # Use your ResNet CIFAR implementations
            architecture_map = {
                "resnet20": resnet20_cifar,
                "resnet32": resnet32_cifar,
                "resnet56": resnet56_cifar,
            }
            
            if architecture not in architecture_map:
                raise ValueError(f"Architecture '{architecture}' not supported")
            
            model = architecture_map[architecture](num_classes=num_classes, **kwargs)
            
            # Load checkpoint if provided
            if checkpoint_path and os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            
            return model
            
        elif model_type == "custom_vision" and CUSTOM_AVAILABLE:
            # Use your CustomVisionClassifier
            model = CustomVisionClassifier(
                num_classes=num_classes,
                **kwargs
            )
            
            if checkpoint_path and os.path.exists(checkpoint_path):
                model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
            
            return model
            
        else:
            raise ValueError(f"Model type '{model_type}' not available or not implemented")
    
    @staticmethod
    def load_model(
        model_path: str,
        model_class: str,
        device: str = "cpu",
        **model_kwargs
    ) -> nn.Module:
        """
        Load a saved model using your actual model classes.
        
        Args:
            model_path: Path to the saved model
            model_class: Name of your model class
            device: Device to load the model on
            **model_kwargs: Arguments for model initialization
            
        Returns:
            Loaded model instance
        """
        
        # Map model class names to actual classes
        model_class_map = {}
        
        if BERT_AVAILABLE:
            model_class_map['BERTSentimentClassifier'] = BERTSentimentClassifier
            
        if RESNET_AVAILABLE:
            model_class_map['ResNetCIFAR'] = ResNetCIFAR
            
        if CUSTOM_AVAILABLE:
            model_class_map['CustomTextClassifier'] = CustomTextClassifier
            model_class_map['CustomVisionClassifier'] = CustomVisionClassifier
        
        if model_class not in model_class_map:
            raise ValueError(f"Model class '{model_class}' not found or not available")
        
        # Create model instance
        ModelClass = model_class_map[model_class]
        model = ModelClass(**model_kwargs)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model.to(device)
    
    @staticmethod
    def get_available_models() -> Dict[str, list]:
        """Get list of available model implementations."""
        available = {
            'text_models': [],
            'vision_models': [],
            'custom_models': []
        }
        
        if BERT_AVAILABLE:
            available['text_models'].append('bert_sentiment')
            
        if RESNET_AVAILABLE:
            available['vision_models'].extend(['resnet20', 'resnet32', 'resnet56'])
            
        if CUSTOM_AVAILABLE:
            available['custom_models'].extend(['custom_text', 'custom_vision'])
        
        return available


# Convenience functions that work with your models
def create_bert_model(model_name: str = "bert-base-uncased", **kwargs):
    """Convenience function to create your BERT sentiment model."""
    return ModelFactory.create_text_model(
        model_name=model_name,
        model_type="bert_sentiment",
        **kwargs
    )


def create_resnet56_model(num_classes: int = 10, **kwargs):
    """Convenience function to create your ResNet-56 CIFAR model."""
    return ModelFactory.create_vision_model(
        architecture="resnet56",
        num_classes=num_classes,
        model_type="resnet_cifar",
        **kwargs
    )
