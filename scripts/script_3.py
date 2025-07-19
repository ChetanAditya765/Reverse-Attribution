# 3. Model Utilities Implementation
model_utils_code = '''"""
Model Utilities for Reverse Attribution Framework
Provides unified interfaces for different model types and frameworks
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from abc import ABC, abstractmethod
import warnings
import logging

class BaseModelWrapper(ABC):
    """Base class for all model wrappers."""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.logger = logging.getLogger(__name__)
        self._setup_model()
    
    def _setup_model(self):
        """Setup model for inference."""
        if hasattr(self.model, 'eval'):
            self.model.eval()
        if hasattr(self.model, 'to') and self.device != 'cpu':
            self.model.to(self.device)
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the wrapped model."""
        info = {
            'model_type': self.__class__.__name__,
            'framework': self._detect_framework(),
            'device': self.device,
            'trainable_params': self._count_parameters()
        }
        
        if hasattr(self, 'num_classes'):
            info['num_classes'] = self.num_classes
        if hasattr(self, 'input_size'):
            info['input_size'] = self.input_size
            
        return info
    
    def _detect_framework(self) -> str:
        """Detect the ML framework used."""
        if hasattr(self.model, 'parameters'):
            return 'pytorch'
        elif hasattr(self.model, 'predict_proba'):
            return 'sklearn'
        elif hasattr(self.model, 'call'):
            return 'tensorflow'
        else:
            return 'unknown'
    
    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        if hasattr(self.model, 'parameters'):
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        else:
            return 0

class SklearnModelWrapper(BaseModelWrapper):
    """Wrapper for scikit-learn models."""
    
    def __init__(self, model, feature_names=None):
        super().__init__(model)
        self.feature_names = feature_names
        self.num_classes = self._get_num_classes()
        
        # Store input size if available
        if hasattr(model, 'n_features_in_'):
            self.input_size = model.n_features_in_
        elif feature_names:
            self.input_size = len(feature_names)
    
    def _get_num_classes(self):
        """Get number of classes from the model."""
        if hasattr(self.model, 'classes_'):
            return len(self.model.classes_)
        elif hasattr(self.model, 'n_classes_'):
            return self.model.n_classes_
        else:
            return None
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        elif hasattr(self.model, 'decision_function'):
            # For SVM and other models that don't have predict_proba
            decision = self.model.decision_function(X)
            if decision.ndim == 1:
                # Binary classification
                decision = decision.reshape(-1, 1)
                decision = np.hstack([-decision, decision])
            # Convert to probabilities using softmax
            exp_scores = np.exp(decision - np.max(decision, axis=1, keepdims=True))
            return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        else:
            raise ValueError("Model does not support probability prediction")

class PyTorchModelWrapper(BaseModelWrapper):
    """Wrapper for PyTorch models."""
    
    def __init__(self, model, device='cpu', input_size=None, num_classes=None, 
                 preprocessing_fn=None, postprocessing_fn=None):
        super().__init__(model, device)
        self.input_size = input_size
        self.num_classes = num_classes
        self.preprocessing_fn = preprocessing_fn
        self.postprocessing_fn = postprocessing_fn
        
        # Try to infer num_classes from model
        if self.num_classes is None:
            self.num_classes = self._infer_num_classes()
    
    def _infer_num_classes(self):
        """Try to infer number of classes from model architecture."""
        try:
            # Look for the last linear layer
            for module in reversed(list(self.model.modules())):
                if isinstance(module, nn.Linear):
                    return module.out_features
            return None
        except:
            return None
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        self.model.eval()
        
        with torch.no_grad():
            # Preprocessing
            if self.preprocessing_fn:
                X = self.preprocessing_fn(X)
            
            # Convert to tensor
            if not isinstance(X, torch.Tensor):
                X = torch.FloatTensor(X)
            
            X = X.to(self.device)
            
            # Forward pass
            logits = self.model(X)
            
            # Apply softmax to get probabilities
            probas = F.softmax(logits, dim=1)
            
            # Postprocessing
            probas_np = probas.cpu().numpy()
            if self.postprocessing_fn:
                probas_np = self.postprocessing_fn(probas_np)
            
            return probas_np

class HuggingFaceModelWrapper(BaseModelWrapper):
    """Wrapper for HuggingFace Transformers models."""
    
    def __init__(self, model, tokenizer, max_length=512, device='cpu'):
        super().__init__(model, device)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_classes = self._get_num_classes()
        
    def _get_num_classes(self):
        """Get number of classes from model config."""
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'num_labels'):
            return self.model.config.num_labels
        elif hasattr(self.model, 'classifier') and hasattr(self.model.classifier, 'out_features'):
            return self.model.classifier.out_features
        else:
            return None
    
    def predict(self, X: Union[List[str], np.ndarray]) -> np.ndarray:
        """Predict class labels."""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
    def predict_proba(self, X: Union[List[str], np.ndarray]) -> np.ndarray:
        """Predict class probabilities."""
        self.model.eval()
        
        # Handle different input types
        if isinstance(X, np.ndarray):
            if X.dtype == object or X.dtype.kind == 'U':
                # String array
                texts = X.tolist()
            else:
                # Assume already tokenized
                return self._predict_from_tokens(X)
        else:
            texts = X
        
        # Tokenize texts
        with torch.no_grad():
            encoding = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Apply softmax
            probas = F.softmax(logits, dim=1)
            
            return probas.cpu().numpy()
    
    def _predict_from_tokens(self, X: np.ndarray) -> np.ndarray:
        """Predict from already tokenized input."""
        self.model.eval()
        
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X = torch.LongTensor(X)
            
            X = X.to(self.device)
            
            # Create attention mask (assuming non-zero tokens are valid)
            attention_mask = (X != 0).float()
            
            outputs = self.model(input_ids=X, attention_mask=attention_mask)
            logits = outputs.logits
            
            probas = F.softmax(logits, dim=1)
            
            return probas.cpu().numpy()
    
    def tokenize_text(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize input texts."""
        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

class TensorFlowModelWrapper(BaseModelWrapper):
    """Wrapper for TensorFlow/Keras models."""
    
    def __init__(self, model, preprocessing_fn=None):
        super().__init__(model)
        self.preprocessing_fn = preprocessing_fn
        self.num_classes = self._get_num_classes()
    
    def _get_num_classes(self):
        """Get number of classes from model."""
        try:
            if hasattr(self.model, 'output_shape'):
                return self.model.output_shape[-1]
            elif hasattr(self.model, 'layers') and len(self.model.layers) > 0:
                last_layer = self.model.layers[-1]
                if hasattr(last_layer, 'units'):
                    return last_layer.units
            return None
        except:
            return None
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.preprocessing_fn:
            X = self.preprocessing_fn(X)
        
        predictions = self.model.predict(X, verbose=0)
        
        # Ensure probabilities sum to 1
        if predictions.shape[1] > 1:
            predictions = predictions / np.sum(predictions, axis=1, keepdims=True)
        
        return predictions

class CustomModelWrapper(BaseModelWrapper):
    """Wrapper for custom models with user-defined prediction functions."""
    
    def __init__(self, predict_fn: Callable, predict_proba_fn: Callable = None, 
                 num_classes: int = None, model_name: str = "custom"):
        self.predict_fn = predict_fn
        self.predict_proba_fn = predict_proba_fn
        self.num_classes = num_classes
        self.model_name = model_name
        self.model = None  # No underlying model
        self.device = 'cpu'
        self.logger = logging.getLogger(__name__)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.predict_fn(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.predict_proba_fn:
            return self.predict_proba_fn(X)
        else:
            # Try to derive from predictions if possible
            predictions = self.predict(X)
            if self.num_classes:
                # Create one-hot encoded probabilities
                probas = np.zeros((len(predictions), self.num_classes))
                probas[np.arange(len(predictions)), predictions] = 1.0
                return probas
            else:
                raise ValueError("predict_proba_fn not provided and num_classes not specified")

class ModelFactory:
    """Factory for creating model wrappers."""
    
    @staticmethod
    def create_wrapper(model, wrapper_type: str = 'auto', **kwargs) -> BaseModelWrapper:
        """
        Create appropriate model wrapper.
        
        Args:
            model: The model to wrap
            wrapper_type: Type of wrapper ('auto', 'sklearn', 'pytorch', 'huggingface', 'tensorflow', 'custom')
            **kwargs: Additional arguments for wrapper
            
        Returns:
            Model wrapper instance
        """
        if wrapper_type == 'auto':
            wrapper_type = ModelFactory._detect_model_type(model)
        
        if wrapper_type == 'sklearn':
            return SklearnModelWrapper(model, **kwargs)
        elif wrapper_type == 'pytorch':
            return PyTorchModelWrapper(model, **kwargs)
        elif wrapper_type == 'huggingface':
            return HuggingFaceModelWrapper(model, **kwargs)
        elif wrapper_type == 'tensorflow':
            return TensorFlowModelWrapper(model, **kwargs)
        elif wrapper_type == 'custom':
            return CustomModelWrapper(model, **kwargs)
        else:
            raise ValueError(f"Unknown wrapper type: {wrapper_type}")
    
    @staticmethod
    def _detect_model_type(model) -> str:
        """Automatically detect model type."""
        # Check for PyTorch
        if hasattr(model, 'parameters') and hasattr(model, 'forward'):
            return 'pytorch'
        
        # Check for scikit-learn
        elif hasattr(model, 'predict') and hasattr(model, 'fit'):
            return 'sklearn'
        
        # Check for TensorFlow/Keras
        elif hasattr(model, 'call') or str(type(model)).find('tensorflow') != -1:
            return 'tensorflow'
        
        # Check for HuggingFace (has config attribute)
        elif hasattr(model, 'config') and hasattr(model, 'forward'):
            return 'huggingface'
        
        else:
            return 'custom'

class EnsembleModelWrapper(BaseModelWrapper):
    """Wrapper for ensemble of multiple models."""
    
    def __init__(self, models: List[BaseModelWrapper], aggregation: str = 'mean',
                 weights: List[float] = None):
        self.models = models
        self.aggregation = aggregation
        self.weights = weights or [1.0] * len(models)
        self.num_classes = self._get_num_classes()
        self.device = 'cpu'
        self.logger = logging.getLogger(__name__)
    
    def _get_num_classes(self):
        """Get number of classes from ensemble models."""
        for model in self.models:
            if hasattr(model, 'num_classes') and model.num_classes:
                return model.num_classes
        return None
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using ensemble."""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using ensemble."""
        predictions = []
        
        for model, weight in zip(self.models, self.weights):
            probas = model.predict_proba(X)
            predictions.append(probas * weight)
        
        predictions = np.array(predictions)
        
        if self.aggregation == 'mean':
            return np.mean(predictions, axis=0)
        elif self.aggregation == 'max':
            return np.max(predictions, axis=0)
        elif self.aggregation == 'min':
            return np.min(predictions, axis=0)
        elif self.aggregation == 'median':
            return np.median(predictions, axis=0)
        else:
            return np.mean(predictions, axis=0)

# Utility functions
def get_model_complexity(model_wrapper: BaseModelWrapper) -> Dict[str, Any]:
    """Estimate model complexity metrics."""
    complexity = {}
    
    # Parameter count
    complexity['num_parameters'] = model_wrapper._count_parameters()
    
    # Input/output dimensions
    if hasattr(model_wrapper, 'input_size'):
        complexity['input_size'] = model_wrapper.input_size
    if hasattr(model_wrapper, 'num_classes'):
        complexity['output_size'] = model_wrapper.num_classes
    
    # Model type
    complexity['framework'] = model_wrapper._detect_framework()
    
    # Memory estimation (rough)
    if complexity['num_parameters'] > 0:
        complexity['estimated_memory_mb'] = complexity['num_parameters'] * 4 / (1024**2)  # 4 bytes per float32
    
    return complexity

def benchmark_model_speed(model_wrapper: BaseModelWrapper, X_sample: np.ndarray, 
                         num_runs: int = 100) -> Dict[str, float]:
    """Benchmark model inference speed."""
    import time
    
    # Warmup
    for _ in range(5):
        _ = model_wrapper.predict_proba(X_sample[:1])
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        _ = model_wrapper.predict_proba(X_sample[:1])
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        'mean_inference_time': np.mean(times),
        'std_inference_time': np.std(times),
        'median_inference_time': np.median(times),
        'min_inference_time': np.min(times),
        'max_inference_time': np.max(times)
    }

def validate_model_wrapper(model_wrapper: BaseModelWrapper, X_test: np.ndarray) -> bool:
    """Validate that model wrapper works correctly."""
    try:
        # Test predict
        predictions = model_wrapper.predict(X_test[:2])
        assert len(predictions) == 2
        
        # Test predict_proba
        probas = model_wrapper.predict_proba(X_test[:2])
        assert probas.shape[0] == 2
        assert np.allclose(np.sum(probas, axis=1), 1.0)  # Probabilities should sum to 1
        
        # Test consistency
        pred_from_probas = np.argmax(probas, axis=1)
        assert np.array_equal(predictions, pred_from_probas)
        
        return True
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Model wrapper validation failed: {str(e)}")
        return False
'''

# Write the model utilities
with open('reverse-attribution/core/model_utils.py', 'w') as f:
    f.write(model_utils_code)

print("✅ Model utilities implementation completed!")
print("Features implemented:")
print("- Unified wrappers for scikit-learn, PyTorch, HuggingFace, TensorFlow models")
print("- Automatic model type detection")
print("- CustomModelWrapper for user-defined prediction functions")
print("- EnsembleModelWrapper for model ensembles")
print("- Model complexity estimation and speed benchmarking")
print("- Model wrapper validation utilities")
print("- Support for both classification and probability prediction")