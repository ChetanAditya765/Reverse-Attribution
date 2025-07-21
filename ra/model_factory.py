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

# Import unified model checking system
from ra.model_utils import unified_model_check, validate_model_for_training
import warnings

# Suppress transformers warnings that clutter output
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", message="resume_download is deprecated")

# Silent model imports - no contradictory warnings
def _import_models():
    """Import models silently - availability checked by unified system."""
    models = {}
    
    try:
        from models.bert_sentiment import BERTSentimentClassifier, create_bert_sentiment_model
        models['bert'] = {'BERTSentimentClassifier': BERTSentimentClassifier, 
                         'create_bert_sentiment_model': create_bert_sentiment_model}
    except ImportError:
        pass
    
    try:
        from models.resnet_cifar import ResNetCIFAR, resnet56_cifar, resnet20_cifar, resnet32_cifar
        models['resnet'] = {'ResNetCIFAR': ResNetCIFAR, 'resnet56_cifar': resnet56_cifar,
                           'resnet20_cifar': resnet20_cifar, 'resnet32_cifar': resnet32_cifar}
    except ImportError:
        pass
    
    try:
        from models.custom_model_example import CustomTextClassifier, CustomVisionClassifier
        models['custom'] = {'CustomTextClassifier': CustomTextClassifier, 
                           'CustomVisionClassifier': CustomVisionClassifier}
    except ImportError:
        pass
    
    return models

_MODELS = _import_models()



class ModelFactory:
    """
    Factory class that creates instances of your actual model implementations.
    """
    @staticmethod  
    def _ensure_status_reported():
        """Ensure status is reported exactly once."""
        from ra.model_utils import ensure_initialized
        ensure_initialized()
    
    @staticmethod
    def create_text_model(model_type: str = "bert_sentiment", **kwargs):
        """Create text model with controlled status reporting."""
        ModelFactory._ensure_status_reported()  # Called once per session
        
        # Rest of your existing code...
        validate_model_for_training(model_type)
    @staticmethod
    def create_text_model(
        model_name: str = "bert-base-uncased",
        num_classes: int = 2,
        checkpoint_path: Optional[str] = None,
        model_type: str = "bert_sentiment",
        **kwargs
    ) -> nn.Module:
        """Create text model with unified validation."""
    
        # Validate availability using unified system
        validate_model_for_training(model_type)
    
        if model_type == "bert_sentiment":
            if 'bert' not in _MODELS:
                raise RuntimeError("BERT sentiment model classes not available")
        
            BERTSentimentClassifier = _MODELS['bert']['BERTSentimentClassifier']
            model = BERTSentimentClassifier(model_name=model_name, num_classes=num_classes, **kwargs)
        
            # Load checkpoint if provided
            if checkpoint_path and os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
        
            return model
    
        elif model_type == "custom_text":
            if 'custom' not in _MODELS:
                raise RuntimeError("Custom text model classes not available")
            
            CustomTextClassifier = _MODELS['custom']['CustomTextClassifier']
            model = CustomTextClassifier(num_classes=num_classes, **kwargs)
        
            if checkpoint_path and os.path.exists(checkpoint_path):
                model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        
            return model
    
        else:
            raise ValueError(f"Model type '{model_type}' not available or not implemented")

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
        """Create vision model with unified validation."""
    
        # Validate availability using unified system
        validate_model_for_training(model_type)
    
        if model_type == "resnet_cifar":
            if 'resnet' not in _MODELS:
                raise RuntimeError("ResNet CIFAR model classes not available")
        
            architecture_map = {
                "resnet20": _MODELS['resnet']['resnet20_cifar'],
                "resnet32": _MODELS['resnet']['resnet32_cifar'],
                "resnet56": _MODELS['resnet']['resnet56_cifar'],
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
    
        elif model_type == "custom_vision":
            if 'custom' not in _MODELS:
                raise RuntimeError("Custom vision model classes not available")
            
            CustomVisionClassifier = _MODELS['custom']['CustomVisionClassifier']
            model = CustomVisionClassifier(num_classes=num_classes, **kwargs)
        
            if checkpoint_path and os.path.exists(checkpoint_path):
                model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        
            return model
    
        else:
            raise ValueError(f"Model type '{model_type}' not available or not implemented")

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
    Load a saved model using unified model checking system.
    
    Args:
        model_path: Path to the saved model
        model_class: Name of your model class
        device: Device to load the model on
        **model_kwargs: Arguments for model initialization
        
    Returns:
        Loaded model instance
    """
    from ra.model_utils import unified_model_check
    
    # Map model class names to model types for validation
    class_to_model_type = {
        'BERTSentimentClassifier': 'bert_sentiment',
        'ResNetCIFAR': 'resnet_cifar',
        'CustomTextClassifier': 'custom_models',
        'CustomVisionClassifier': 'custom_models'
    }
    
    if model_class not in class_to_model_type:
        raise ValueError(f"Unknown model class '{model_class}'. Supported classes: {list(class_to_model_type.keys())}")
    
    # Validate model availability using unified system
    model_type = class_to_model_type[model_class]
    status = unified_model_check(model_type)
    
    if not status['available']:
        raise ValueError(f"Model class '{model_class}' not available. Error: {status['error']}")
    
    # Build model class mapping only for available models
    model_class_map = {}
    
    # Import classes only if they're available
    if model_class in ['BERTSentimentClassifier'] and 'bert' in _MODELS:
        model_class_map['BERTSentimentClassifier'] = _MODELS['bert']['BERTSentimentClassifier']
    
    if model_class in ['ResNetCIFAR'] and 'resnet' in _MODELS:
        model_class_map['ResNetCIFAR'] = _MODELS['resnet']['ResNetCIFAR']
    
    if model_class in ['CustomTextClassifier', 'CustomVisionClassifier'] and 'custom' in _MODELS:
        model_class_map['CustomTextClassifier'] = _MODELS['custom']['CustomTextClassifier']
        model_class_map['CustomVisionClassifier'] = _MODELS['custom']['CustomVisionClassifier']
    
    if model_class not in model_class_map:
        raise ValueError(f"Model class '{model_class}' not found in available implementations")
    
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
    """Get list of available model implementations using unified checking."""
    from ra.model_utils import list_available_models
    
    available = {
        'text_models': [],
        'vision_models': [],
        'custom_models': []
    }
    
    # Get available models from unified system
    available_model_names = list_available_models()
    
    # Categorize available models
    for model_name in available_model_names:
        if model_name == 'bert_sentiment':
            available['text_models'].append('bert_sentiment')
        elif model_name == 'resnet_cifar':
            available['vision_models'].extend(['resnet20', 'resnet32', 'resnet56'])
        elif model_name == 'custom_models':
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
