# 2. Explainer Utilities Implementation
explainer_utils_code = '''"""
Explainer Utilities for Reverse Attribution Framework
Provides unified interfaces for SHAP, Integrated Gradients, LIME, and Anchors
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from abc import ABC, abstractmethod
import warnings
import logging

class BaseExplainer(ABC):
    """Base class for all explainer implementations."""
    
    def __init__(self, model_wrapper):
        self.model_wrapper = model_wrapper
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def explain(self, x: np.ndarray, target_class: int, **kwargs) -> np.ndarray:
        """Compute attributions for the given input and target class."""
        pass

class SHAPExplainer(BaseExplainer):
    """SHAP explainer with support for different explainer types."""
    
    def __init__(self, model_wrapper, explainer_type='kernel', background_data=None, **kwargs):
        super().__init__(model_wrapper)
        self.explainer_type = explainer_type
        self.background_data = background_data
        self.explainer = None
        self._setup_explainer(**kwargs)
    
    def _setup_explainer(self, **kwargs):
        """Initialize the appropriate SHAP explainer."""
        try:
            import shap
            
            if self.explainer_type == 'kernel':
                # Model-agnostic explainer
                def predict_fn(X):
                    return self.model_wrapper.predict_proba(X)
                
                background = self.background_data if self.background_data is not None else np.zeros((1, 10))
                self.explainer = shap.KernelExplainer(predict_fn, background)
                
            elif self.explainer_type == 'tree':
                # For tree-based models
                if hasattr(self.model_wrapper.model, 'estimators_'):
                    self.explainer = shap.TreeExplainer(self.model_wrapper.model)
                else:
                    self.logger.warning("Tree explainer requested but model is not tree-based. Falling back to kernel.")
                    self._setup_kernel_explainer()
                    
            elif self.explainer_type == 'gradient':
                # For deep learning models
                if hasattr(self.model_wrapper.model, 'parameters'):
                    background = torch.zeros((1, self.model_wrapper.input_size)) if hasattr(self.model_wrapper, 'input_size') else None
                    self.explainer = shap.GradientExplainer(self.model_wrapper.model, background)
                else:
                    self.logger.warning("Gradient explainer requested but model is not PyTorch. Falling back to kernel.")
                    self._setup_kernel_explainer()
                    
            elif self.explainer_type == 'deep':
                # DeepExplainer for neural networks
                if hasattr(self.model_wrapper.model, 'parameters'):
                    background = torch.zeros((1, self.model_wrapper.input_size)) if hasattr(self.model_wrapper, 'input_size') else None
                    self.explainer = shap.DeepExplainer(self.model_wrapper.model, background)
                else:
                    self.logger.warning("Deep explainer requested but model is not PyTorch. Falling back to kernel.")
                    self._setup_kernel_explainer()
            else:
                self._setup_kernel_explainer()
                
        except ImportError:
            self.logger.error("SHAP library not available. Please install with: pip install shap")
            raise ImportError("SHAP library is required for SHAP explanations")
    
    def _setup_kernel_explainer(self):
        """Setup kernel explainer as fallback."""
        import shap
        def predict_fn(X):
            return self.model_wrapper.predict_proba(X)
        
        background = self.background_data if self.background_data is not None else np.zeros((1, 10))
        self.explainer = shap.KernelExplainer(predict_fn, background)
    
    def explain(self, x: np.ndarray, target_class: int = None, nsamples: int = 100) -> np.ndarray:
        """
        Compute SHAP attributions.
        
        Args:
            x: Input instance
            target_class: Target class (if None, use predicted class)
            nsamples: Number of samples for kernel explainer
            
        Returns:
            Attribution values
        """
        if self.explainer is None:
            raise RuntimeError("SHAP explainer not properly initialized")
        
        x_input = x.reshape(1, -1) if len(x.shape) == 1 else x
        
        try:
            if self.explainer_type == 'gradient' or self.explainer_type == 'deep':
                # Handle PyTorch models
                if isinstance(x_input, np.ndarray):
                    x_tensor = torch.FloatTensor(x_input)
                else:
                    x_tensor = x_input
                
                shap_values = self.explainer.shap_values(x_tensor)
                
                # Handle multi-class output
                if isinstance(shap_values, list):
                    if target_class is not None:
                        return shap_values[target_class][0]
                    else:
                        # Return for predicted class
                        probs = self.model_wrapper.predict_proba(x_input)
                        pred_class = np.argmax(probs[0])
                        return shap_values[pred_class][0]
                else:
                    return shap_values[0]
                    
            else:
                # Handle kernel and tree explainers
                shap_values = self.explainer.shap_values(x_input, nsamples=nsamples)
                
                if isinstance(shap_values, list):
                    if target_class is not None:
                        return shap_values[target_class][0]
                    else:
                        # Return for predicted class
                        probs = self.model_wrapper.predict_proba(x_input)
                        pred_class = np.argmax(probs[0])
                        return shap_values[pred_class][0]
                else:
                    return shap_values[0]
                    
        except Exception as e:
            self.logger.error(f"Error computing SHAP values: {str(e)}")
            # Fallback to numerical gradients
            return self._numerical_gradients(x, target_class)
    
    def _numerical_gradients(self, x: np.ndarray, target_class: int) -> np.ndarray:
        """Fallback numerical gradient computation."""
        h = 1e-5
        grad = np.zeros_like(x)
        
        if target_class is None:
            probs = self.model_wrapper.predict_proba(x.reshape(1, -1))
            target_class = np.argmax(probs[0])
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            
            prob_plus = self.model_wrapper.predict_proba(x_plus.reshape(1, -1))[0, target_class]
            prob_minus = self.model_wrapper.predict_proba(x_minus.reshape(1, -1))[0, target_class]
            
            grad[i] = (prob_plus - prob_minus) / (2 * h)
            
        return grad

class IntegratedGradientsExplainer(BaseExplainer):
    """Integrated Gradients explainer using Captum library."""
    
    def __init__(self, model_wrapper, baseline_strategy='zero', n_steps=50):
        super().__init__(model_wrapper)
        self.baseline_strategy = baseline_strategy
        self.n_steps = n_steps
        self.explainer = None
        self._setup_explainer()
    
    def _setup_explainer(self):
        """Initialize Integrated Gradients explainer."""
        try:
            from captum.attr import IntegratedGradients
            
            if hasattr(self.model_wrapper.model, 'parameters'):
                # PyTorch model
                self.explainer = IntegratedGradients(self.model_wrapper.model)
            else:
                # Use manual implementation for non-PyTorch models
                self.explainer = None
                self.logger.info("Using manual IG implementation for non-PyTorch model")
                
        except ImportError:
            self.logger.warning("Captum not available. Using manual IG implementation.")
            self.explainer = None
    
    def explain(self, x: np.ndarray, target_class: int = None, baseline: np.ndarray = None) -> np.ndarray:
        """
        Compute Integrated Gradients attributions.
        
        Args:
            x: Input instance
            target_class: Target class
            baseline: Baseline input
            
        Returns:
            Attribution values
        """
        if baseline is None:
            baseline = self._compute_baseline(x)
        
        if target_class is None:
            probs = self.model_wrapper.predict_proba(x.reshape(1, -1))
            target_class = np.argmax(probs[0])
        
        if self.explainer is not None and hasattr(self.model_wrapper.model, 'parameters'):
            # Use Captum implementation
            return self._captum_ig(x, target_class, baseline)
        else:
            # Use manual implementation
            return self._manual_ig(x, target_class, baseline)
    
    def _compute_baseline(self, x: np.ndarray) -> np.ndarray:
        """Compute baseline based on strategy."""
        if self.baseline_strategy == 'zero':
            return np.zeros_like(x)
        elif self.baseline_strategy == 'mean':
            return np.mean(x) * np.ones_like(x)
        elif self.baseline_strategy == 'uniform':
            return np.random.uniform(0, 1, x.shape)
        else:
            return np.zeros_like(x)
    
    def _captum_ig(self, x: np.ndarray, target_class: int, baseline: np.ndarray) -> np.ndarray:
        """Use Captum's Integrated Gradients implementation."""
        x_tensor = torch.FloatTensor(x).unsqueeze(0).requires_grad_(True)
        baseline_tensor = torch.FloatTensor(baseline).unsqueeze(0)
        
        attributions = self.explainer.attribute(
            x_tensor, 
            baseline_tensor, 
            target=target_class,
            n_steps=self.n_steps
        )
        
        return attributions.detach().numpy()[0]
    
    def _manual_ig(self, x: np.ndarray, target_class: int, baseline: np.ndarray) -> np.ndarray:
        """Manual Integrated Gradients implementation."""
        # Generate path from baseline to input
        alphas = np.linspace(0, 1, self.n_steps)
        gradients = []
        
        for alpha in alphas:
            x_interp = baseline + alpha * (x - baseline)
            grad = self._compute_gradients(x_interp, target_class)
            gradients.append(grad)
        
        # Average gradients and multiply by input difference
        avg_gradients = np.mean(gradients, axis=0)
        attributions = (x - baseline) * avg_gradients
        
        return attributions
    
    def _compute_gradients(self, x: np.ndarray, target_class: int) -> np.ndarray:
        """Compute gradients numerically."""
        h = 1e-5
        grad = np.zeros_like(x)
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            
            prob_plus = self.model_wrapper.predict_proba(x_plus.reshape(1, -1))[0, target_class]
            prob_minus = self.model_wrapper.predict_proba(x_minus.reshape(1, -1))[0, target_class]
            
            grad[i] = (prob_plus - prob_minus) / (2 * h)
            
        return grad

class LIMEExplainer(BaseExplainer):
    """LIME explainer for tabular data."""
    
    def __init__(self, model_wrapper, training_data=None, feature_names=None, **kwargs):
        super().__init__(model_wrapper)
        self.training_data = training_data
        self.feature_names = feature_names
        self.explainer = None
        self.lime_kwargs = kwargs
        self._setup_explainer()
    
    def _setup_explainer(self):
        """Initialize LIME explainer."""
        try:
            from lime.lime_tabular import LimeTabularExplainer
            
            # Use provided training data or create dummy data
            if self.training_data is not None:
                training_data = self.training_data
            else:
                # Create minimal training data
                training_data = np.random.normal(0, 1, (100, 10))  # Default shape
                self.logger.warning("No training data provided for LIME. Using random data.")
            
            feature_names = self.feature_names or [f'feature_{i}' for i in range(training_data.shape[1])]
            
            self.explainer = LimeTabularExplainer(
                training_data=training_data,
                feature_names=feature_names,
                discretize_continuous=self.lime_kwargs.get('discretize_continuous', False),
                **{k: v for k, v in self.lime_kwargs.items() if k != 'discretize_continuous'}
            )
            
        except ImportError:
            self.logger.error("LIME library not available. Please install with: pip install lime")
            raise ImportError("LIME library is required for LIME explanations")
    
    def explain(self, x: np.ndarray, target_class: int = None, num_features: int = None) -> np.ndarray:
        """
        Compute LIME attributions.
        
        Args:
            x: Input instance
            target_class: Target class
            num_features: Number of features to include in explanation
            
        Returns:
            Attribution values
        """
        if self.explainer is None:
            raise RuntimeError("LIME explainer not properly initialized")
        
        if target_class is None:
            probs = self.model_wrapper.predict_proba(x.reshape(1, -1))
            target_class = np.argmax(probs[0])
        
        if num_features is None:
            num_features = len(x)
        
        def predict_fn(X):
            return self.model_wrapper.predict_proba(X)
        
        try:
            explanation = self.explainer.explain_instance(
                x, 
                predict_fn, 
                labels=[target_class], 
                num_features=num_features
            )
            
            # Extract attributions
            attributions = np.zeros_like(x)
            feature_importance = explanation.as_list()
            
            for feature_name, importance in feature_importance:
                if 'feature_' in feature_name:
                    feature_idx = int(feature_name.split('_')[-1])
                else:
                    # Try to parse as integer
                    try:
                        feature_idx = int(feature_name)
                    except ValueError:
                        # Skip if can't parse
                        continue
                
                if 0 <= feature_idx < len(attributions):
                    attributions[feature_idx] = importance
            
            return attributions
            
        except Exception as e:
            self.logger.error(f"Error computing LIME explanation: {str(e)}")
            return np.zeros_like(x)

class AnchorsExplainer(BaseExplainer):
    """Anchors explainer for rule-based explanations."""
    
    def __init__(self, model_wrapper, training_data=None, feature_names=None, **kwargs):
        super().__init__(model_wrapper)
        self.training_data = training_data
        self.feature_names = feature_names
        self.explainer = None
        self.anchors_kwargs = kwargs
        self._setup_explainer()
    
    def _setup_explainer(self):
        """Initialize Anchors explainer."""
        try:
            from alibi.explainers import AnchorTabular
            
            # Use provided training data or create dummy data
            if self.training_data is not None:
                training_data = self.training_data
            else:
                training_data = np.random.normal(0, 1, (100, 10))
                self.logger.warning("No training data provided for Anchors. Using random data.")
            
            feature_names = self.feature_names or [f'feature_{i}' for i in range(training_data.shape[1])]
            
            self.explainer = AnchorTabular(
                predictor=self.model_wrapper.predict,
                feature_names=feature_names,
                **self.anchors_kwargs
            )
            
            # Fit the explainer
            self.explainer.fit(training_data)
            
        except ImportError:
            self.logger.error("Alibi library not available. Please install with: pip install alibi")
            raise ImportError("Alibi library is required for Anchors explanations")
    
    def explain(self, x: np.ndarray, threshold: float = 0.95) -> Dict[str, Any]:
        """
        Compute Anchors explanation.
        
        Args:
            x: Input instance
            threshold: Confidence threshold for anchors
            
        Returns:
            Dictionary with anchor rules and coverage
        """
        if self.explainer is None:
            raise RuntimeError("Anchors explainer not properly initialized")
        
        try:
            explanation = self.explainer.explain(x.reshape(1, -1), threshold=threshold)
            
            return {
                'anchor_rules': explanation.data.get('anchor', []),
                'precision': explanation.data.get('precision', 0.0),
                'coverage': explanation.data.get('coverage', 0.0),
                'raw_explanation': explanation
            }
            
        except Exception as e:
            self.logger.error(f"Error computing Anchors explanation: {str(e)}")
            return {
                'anchor_rules': [],
                'precision': 0.0,
                'coverage': 0.0,
                'raw_explanation': None
            }

class ExplainerFactory:
    """Factory class for creating explainer instances."""
    
    @staticmethod
    def create_explainer(explainer_type: str, model_wrapper, **kwargs) -> BaseExplainer:
        """
        Create an explainer instance.
        
        Args:
            explainer_type: Type of explainer ('shap', 'ig', 'lime', 'anchors')
            model_wrapper: Model wrapper instance
            **kwargs: Additional arguments for explainer
            
        Returns:
            Explainer instance
        """
        if explainer_type.lower() == 'shap':
            return SHAPExplainer(model_wrapper, **kwargs)
        elif explainer_type.lower() in ['ig', 'integrated_gradients']:
            return IntegratedGradientsExplainer(model_wrapper, **kwargs)
        elif explainer_type.lower() == 'lime':
            return LIMEExplainer(model_wrapper, **kwargs)
        elif explainer_type.lower() == 'anchors':
            return AnchorsExplainer(model_wrapper, **kwargs)
        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")

class MultiExplainer:
    """Wrapper that runs multiple explainers and aggregates results."""
    
    def __init__(self, explainers: List[BaseExplainer]):
        self.explainers = explainers
        self.logger = logging.getLogger(__name__)
    
    def explain(self, x: np.ndarray, target_class: int = None, 
                aggregation: str = 'mean') -> Dict[str, np.ndarray]:
        """
        Run multiple explainers and aggregate results.
        
        Args:
            x: Input instance
            target_class: Target class
            aggregation: Aggregation method ('mean', 'median', 'max', 'min')
            
        Returns:
            Dictionary with individual and aggregated attributions
        """
        results = {}
        attributions = []
        
        for i, explainer in enumerate(self.explainers):
            try:
                attr = explainer.explain(x, target_class)
                explainer_name = explainer.__class__.__name__
                results[f'{explainer_name}_{i}'] = attr
                attributions.append(attr)
            except Exception as e:
                self.logger.error(f"Error with explainer {explainer.__class__.__name__}: {str(e)}")
        
        # Aggregate results
        if attributions:
            attributions = np.array(attributions)
            
            if aggregation == 'mean':
                results['aggregated'] = np.mean(attributions, axis=0)
            elif aggregation == 'median':
                results['aggregated'] = np.median(attributions, axis=0)
            elif aggregation == 'max':
                results['aggregated'] = np.max(attributions, axis=0)
            elif aggregation == 'min':
                results['aggregated'] = np.min(attributions, axis=0)
            else:
                results['aggregated'] = np.mean(attributions, axis=0)
        
        return results

# Utility functions for explainer management
def get_available_explainers() -> Dict[str, bool]:
    """Check which explainer libraries are available."""
    availability = {}
    
    try:
        import shap
        availability['shap'] = True
    except ImportError:
        availability['shap'] = False
    
    try:
        import captum
        availability['captum'] = True
    except ImportError:
        availability['captum'] = False
    
    try:
        import lime
        availability['lime'] = True
    except ImportError:
        availability['lime'] = False
    
    try:
        import alibi
        availability['alibi'] = True
    except ImportError:
        availability['alibi'] = False
    
    return availability

def recommend_explainer(model_wrapper) -> str:
    """Recommend the best explainer based on model type and available libraries."""
    availability = get_available_explainers()
    
    # Check model type
    if hasattr(model_wrapper.model, 'parameters'):
        # PyTorch model
        if availability['captum']:
            return 'ig'
        elif availability['shap']:
            return 'shap'
    elif hasattr(model_wrapper.model, 'estimators_'):
        # Tree-based model
        if availability['shap']:
            return 'shap'
    
    # Default fallback
    if availability['shap']:
        return 'shap'
    elif availability['lime']:
        return 'lime'
    else:
        return 'numerical'  # Use numerical gradients as last resort
'''

# Write the explainer utilities
with open('reverse-attribution/core/explainer_utils.py', 'w') as f:
    f.write(explainer_utils_code)

print("✅ Explainer utilities implementation completed!")
print("Features implemented:")
print("- SHAP explainer with multiple types (kernel, tree, gradient, deep)")
print("- Integrated Gradients with Captum integration")
print("- LIME tabular explainer")
print("- Anchors rule-based explainer")
print("- ExplainerFactory for easy instantiation")
print("- MultiExplainer for ensemble explanations")
print("- Automatic library detection and recommendations")