"""
Unified Model Utilities for Reverse Attribution Framework
========================================================

This module provides centralized model availability checking, validation,
and status reporting to eliminate contradictory availability messages
throughout the codebase.

Features:
- Unified model checking across all components
- GPU integration with automatic device detection  
- Checkpoint discovery and validation
- Comprehensive status reporting
- Silent failure handling with detailed diagnostics
"""

import os
import sys
import torch
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
import logging

# Suppress transformers warnings that clutter output
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", message="resume_download is deprecated")

# Setup logging for debugging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class ModelStatus:
    """Comprehensive model status information."""
    name: str
    available: bool
    can_instantiate: bool
    has_checkpoints: bool
    checkpoint_paths: List[str]
    class_importable: bool
    error_message: Optional[str] = None
    device_compatible: bool = True


class UnifiedModelChecker:
    """
    Centralized model checking system that provides consistent availability
    reporting across all components of the Reverse Attribution framework.
    """
    
    def __init__(self):
        # Import device utilities
        try:
            from ra.device_utils import device, get_device
            self.device = device
            self.get_device = get_device
        except ImportError:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.get_device = lambda: self.device
        
        # Add models to path for imports
        self.models_path = Path(__file__).parent.parent / "models"
        if str(self.models_path) not in sys.path:
            sys.path.insert(0, str(self.models_path))
        
        # Define model configurations
        self.model_configs = {
            'bert_sentiment': {
                'module': 'models.bert_sentiment',
                'classes': ['BERTSentimentClassifier'],
                'factory_functions': ['create_bert_sentiment_model'],
                'checkpoint_patterns': [
                    'models/bert_sentiment/',
                    'checkpoints/bert_sentiment/',
                    'saved_models/bert_sentiment/'
                ]
            },
            'resnet_cifar': {
                'module': 'models.resnet_cifar', 
                'classes': ['ResNetCIFAR'],
                'factory_functions': ['resnet56_cifar', 'resnet20_cifar', 'resnet32_cifar'],
                'checkpoint_patterns': [
                    'models/resnet_cifar/',
                    'checkpoints/resnet_cifar/',
                    'saved_models/resnet_cifar/'
                ]
            },
            'custom_models': {
                'module': 'models.custom_model_example',
                'classes': ['CustomTextClassifier', 'CustomVisionClassifier'],
                'factory_functions': [],
                'checkpoint_patterns': [
                    'models/custom/',
                    'checkpoints/custom/',
                    'saved_models/custom/'
                ]
            }
        }
        
        # Cache for model status to avoid repeated checks
        self._status_cache: Dict[str, ModelStatus] = {}
        self._cache_valid = False
    
    def _clear_cache(self):
        """Clear the model status cache."""
        self._status_cache.clear()
        self._cache_valid = False
    
    def _find_checkpoints(self, patterns: List[str]) -> List[str]:
        """Find available checkpoint files matching the given patterns."""
        checkpoints = []
        
        for pattern_dir in patterns:
            pattern_path = Path(pattern_dir)
            if pattern_path.exists() and pattern_path.is_dir():
                # Look for common checkpoint file extensions
                checkpoint_extensions = ['*.pth', '*.pt', '*.bin', '*.ckpt']
                for ext in checkpoint_extensions:
                    checkpoints.extend([str(p) for p in pattern_path.glob(ext)])
        
        return checkpoints
    
    def _test_model_instantiation(self, module_name: str, class_names: List[str]) -> Tuple[bool, Optional[str]]:
        """Test if model classes can be imported and instantiated."""
        try:
            module = __import__(module_name, fromlist=class_names)
            
            for class_name in class_names:
                if hasattr(module, class_name):
                    model_class = getattr(module, class_name)
                    
                    # Try to instantiate with minimal parameters
                    if class_name == 'BERTSentimentClassifier':
                        test_instance = model_class(
                            model_name="bert-base-uncased",
                            num_classes=2
                        )
                    elif class_name == 'ResNetCIFAR':
                        # ResNet needs to be instantiated via factory functions
                        continue
                    else:
                        test_instance = model_class(num_classes=2)
                    
                    # Test device compatibility
                    try:
                        test_instance = test_instance.to(self.device)
                        del test_instance  # Clean up
                        return True, None
                    except Exception as e:
                        return False, f"Device compatibility issue: {str(e)}"
            
            # Test factory functions for ResNet
            if 'resnet' in module_name.lower():
                for func_name in ['resnet56_cifar', 'resnet20_cifar']:
                    if hasattr(module, func_name):
                        factory_func = getattr(module, func_name)
                        test_instance = factory_func(num_classes=10)
                        test_instance = test_instance.to(self.device)
                        del test_instance
                        return True, None
            
            return True, None
            
        except ImportError as e:
            return False, f"Import error: {str(e)}"
        except Exception as e:
            return False, f"Instantiation error: {str(e)}"
    
    def check_model_status(self, model_name: str, force_refresh: bool = False) -> ModelStatus:
        """
        Check comprehensive status of a specific model.
        
        Args:
            model_name: Name of the model to check ('bert_sentiment', 'resnet_cifar', etc.)
            force_refresh: Whether to bypass cache and perform fresh check
            
        Returns:
            ModelStatus object with comprehensive status information
        """
        
        if not force_refresh and model_name in self._status_cache:
            return self._status_cache[model_name]
        
        if model_name not in self.model_configs:
            return ModelStatus(
                name=model_name,
                available=False,
                can_instantiate=False,
                has_checkpoints=False,
                checkpoint_paths=[],
                class_importable=False,
                error_message=f"Unknown model: {model_name}"
            )
        
        config = self.model_configs[model_name]
        
        # Check if classes can be imported and instantiated
        can_instantiate, error_msg = self._test_model_instantiation(
            config['module'], 
            config['classes']
        )
        
        # Find available checkpoints
        checkpoint_paths = self._find_checkpoints(config['checkpoint_patterns'])
        
        # Determine overall availability
        available = can_instantiate  # Model is available if it can be instantiated
        
        status = ModelStatus(
            name=model_name,
            available=available,
            can_instantiate=can_instantiate,
            has_checkpoints=len(checkpoint_paths) > 0,
            checkpoint_paths=checkpoint_paths,
            class_importable=can_instantiate,  # If can instantiate, classes are importable
            error_message=error_msg if not can_instantiate else None,
            device_compatible=True  # Tested during instantiation
        )
        
        # Cache the result
        self._status_cache[model_name] = status
        return status
    
    def check_all_models(self, force_refresh: bool = False) -> Dict[str, ModelStatus]:
        """Check status of all configured models."""
        if force_refresh:
            self._clear_cache()
        
        all_status = {}
        for model_name in self.model_configs.keys():
            all_status[model_name] = self.check_model_status(model_name, force_refresh)
        
        return all_status
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        all_status = self.check_all_models()
        return [name for name, status in all_status.items() if status.available]
    
    def get_models_with_checkpoints(self) -> List[str]:
        """Get list of models that have saved checkpoints."""
        all_status = self.check_all_models()
        return [name for name, status in all_status.items() if status.has_checkpoints]


# Global instance for consistent checking across the codebase
_model_checker = UnifiedModelChecker()


def unified_model_check(model_name: str, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Unified model checking function for use across the codebase.
    
    Args:
        model_name: Name of the model to check
        force_refresh: Whether to bypass cache
        
    Returns:
        Dictionary with model status information
    """
    status = _model_checker.check_model_status(model_name, force_refresh)
    
    return {
        'available': status.available,
        'can_instantiate': status.can_instantiate,
        'has_checkpoints': status.has_checkpoints,
        'checkpoint_paths': status.checkpoint_paths,
        'error': status.error_message,
        'device_compatible': status.device_compatible
    }


def check_model_availability_fixed() -> Dict[str, bool]:
    """
    Fixed model availability checker that provides consistent results.
    Replacement for the original check_model_availability function.
    """
    all_status = _model_checker.check_all_models()
    return {name: status.available for name, status in all_status.items()}


def print_model_status_report(verbose: bool = False):
    """
    Print a comprehensive, unified model status report.
    This eliminates contradictory availability messages.
    
    Args:
        verbose: Whether to include detailed information
    """
    print("🔍 Unified Model Status Report")
    print("=" * 50)
    
    device_info = _model_checker.get_device()
    print(f"🔧 Device: {device_info}")
    
    if torch.cuda.is_available():
        print(f"🔧 GPU: {torch.cuda.get_device_name(0)}")
    
    print("=" * 50)
    
    all_status = _model_checker.check_all_models()
    
    for model_name, status in all_status.items():
        if status.available:
            print(f"✅ {model_name}: Available")
            if verbose:
                print(f"   - Can instantiate: {status.can_instantiate}")
                print(f"   - Has checkpoints: {status.has_checkpoints}")
                if status.checkpoint_paths:
                    print(f"   - Checkpoint paths: {len(status.checkpoint_paths)} found")
        else:
            print(f"❌ {model_name}: Not available")
            if verbose and status.error_message:
                print(f"   - Error: {status.error_message}")
    
    print("=" * 50)
    
    available_models = [name for name, status in all_status.items() if status.available]
    print(f"📊 Summary: {len(available_models)}/{len(all_status)} models available")
    
    if available_models:
        print(f"🎯 Ready for training: {', '.join(available_models)}")


def validate_model_for_training(model_name: str) -> bool:
    """
    Validate that a model is ready for training.
    
    Args:
        model_name: Name of the model to validate
        
    Returns:
        True if model is ready for training, False otherwise
        
    Raises:
        RuntimeError: If model is not available with detailed error message
    """
    status = _model_checker.check_model_status(model_name)
    
    if not status.available:
        error_msg = f"Model '{model_name}' is not available for training."
        if status.error_message:
            error_msg += f" Error: {status.error_message}"
        raise RuntimeError(error_msg)
    
    return True


def get_model_checkpoint_path(model_name: str) -> Optional[str]:
    """
    Get the best available checkpoint path for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Path to the best checkpoint, or None if no checkpoints available
    """
    status = _model_checker.check_model_status(model_name)
    
    if not status.checkpoint_paths:
        return None
    
    # Return the first available checkpoint
    # In a more sophisticated implementation, you could sort by modification date
    return status.checkpoint_paths[0]


def list_available_models() -> List[str]:
    """Get list of available model names."""
    return _model_checker.get_available_models()


def list_models_with_checkpoints() -> List[str]:
    """Get list of models that have saved checkpoints."""
    return _model_checker.get_models_with_checkpoints()


# Initialize and print status when module is imported
print_model_status_report(verbose=False)
