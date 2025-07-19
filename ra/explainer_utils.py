"""
Integration utilities for baseline explanation methods.
Provides unified interface for SHAP, LIME, and other attribution methods.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Optional imports - gracefully handle missing packages
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not available. Install with: pip install shap")

try:
    import lime
    from lime import lime_text, lime_image
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("⚠️  LIME not available. Install with: pip install lime")

try:
    from captum.attr import (
        IntegratedGradients, GradientShap, DeepLift, 
        LayerIntegratedGradients, LayerGradientShap
    )
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    print("⚠️  Captum not available. Install with: pip install captum")


class BaselineExplainer:
    """Base class for baseline explanation methods."""
    
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model.eval()
        self.device = device
        self.model_type = self._detect_model_type()
    
    def _detect_model_type(self) -> str:
        """Detect if model is text, vision, or tabular."""
        if hasattr(self.model, 'embeddings') or hasattr(self.model, 'tokenizer'):
            return "text"
        elif any(isinstance(module, torch.nn.Conv2d) for module in self.model.modules()):
            return "vision"
        else:
            return "tabular"
    
    def explain(self, inputs: Any, method: str = "integrated_gradients", **kwargs) -> Dict[str, Any]:
        """Generic explain method - to be implemented by subclasses."""
        raise NotImplementedError


class SHAPExplainer(BaselineExplainer):
    """SHAP-based explanations."""
    
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        super().__init__(model, device)
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for SHAPExplainer")
        
        self.explainer = None
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize appropriate SHAP explainer based on model type."""
        if self.model_type == "text":
            # For text models, we'll use a wrapper function
            def model_wrapper(texts):
                if hasattr(self.model, 'tokenizer'):
                    encoded = self.model.tokenizer(
                        texts, return_tensors="pt", padding=True, truncation=True
                    )
                    with torch.no_grad():
                        outputs = self.model(encoded['input_ids'].to(self.device))
                        return F.softmax(outputs, dim=-1).cpu().numpy()
                else:
                    # Assume inputs are already tokenized
                    with torch.no_grad():
                        outputs = self.model(torch.tensor(texts).to(self.device))
                        return F.softmax(outputs, dim=-1).cpu().numpy()
            
            self.model_wrapper = model_wrapper
            self.explainer = shap.Explainer(model_wrapper)
            
        elif self.model_type == "vision":
            # For vision models
            def model_wrapper(images):
                if isinstance(images, np.ndarray):
                    images = torch.tensor(images).float()
                with torch.no_grad():
                    outputs = self.model(images.to(self.device))
                    return F.softmax(outputs, dim=-1).cpu().numpy()
            
            self.model_wrapper = model_wrapper
            # Use DeepExplainer for vision models
            try:
                self.explainer = shap.DeepExplainer(self.model, torch.zeros(1, 3, 32, 32).to(self.device))
            except:
                self.explainer = shap.Explainer(model_wrapper)
        
        else:  # tabular
            def model_wrapper(X):
                with torch.no_grad():
                    outputs = self.model(torch.tensor(X).float().to(self.device))
                    return F.softmax(outputs, dim=-1).cpu().numpy()
            
            self.model_wrapper = model_wrapper
            self.explainer = shap.Explainer(model_wrapper)
    
    def explain(
        self, 
        inputs: Union[str, List[str], torch.Tensor, np.ndarray], 
        max_evals: int = 100,
        **kwargs
    ) -> shap.Explanation:
        """
        Generate SHAP explanations.
        
        Args:
            inputs: Input data to explain
            max_evals: Maximum number of model evaluations
            
        Returns:
            SHAP Explanation object
        """
        try:
            if self.model_type == "text" and isinstance(inputs, (str, list)):
                if isinstance(inputs, str):
                    inputs = [inputs]
                return self.explainer(inputs, max_evals=max_evals)
            
            elif self.model_type == "vision":
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.cpu().numpy()
                return self.explainer(inputs)
            
            else:  # tabular
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.cpu().numpy()
                return self.explainer(inputs, max_evals=max_evals)
                
        except Exception as e:
            print(f"SHAP explanation failed: {e}")
            return None


class LIMEExplainer(BaselineExplainer):
    """LIME-based explanations."""
    
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        super().__init__(model, device)
        if not LIME_AVAILABLE:
            raise ImportError("LIME is required for LIMEExplainer")
        
        self.explainer = None
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize appropriate LIME explainer."""
        if self.model_type == "text":
            self.explainer = lime_text.LimeTextExplainer(
                class_names=['negative', 'positive'],  # Default binary classification
                mode='classification'
            )
        elif self.model_type == "vision":
            self.explainer = lime_image.LimeImageExplainer()
        else:  # tabular
            # This requires training data statistics - will be set when explaining
            self.explainer = None
    
    def _model_predict_proba(self, texts):
        """Prediction function for text inputs."""
        if hasattr(self.model, 'tokenizer'):
            encoded = self.model.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            with torch.no_grad():
                outputs = self.model(encoded['input_ids'].to(self.device))
                return F.softmax(outputs, dim=-1).cpu().numpy()
        return None
    
    def _model_predict_image(self, images):
        """Prediction function for image inputs."""
        # Convert to tensor if needed
        if isinstance(images, np.ndarray):
            if len(images.shape) == 3:  # Single image
                images = images[np.newaxis, ...]
            images = torch.tensor(images).permute(0, 3, 1, 2).float()  # NHWC -> NCHW
        
        with torch.no_grad():
            outputs = self.model(images.to(self.device))
            return F.softmax(outputs, dim=-1).cpu().numpy()
    
    def explain(
        self, 
        inputs: Union[str, np.ndarray, torch.Tensor],
        num_features: int = 10,
        num_samples: int = 1000,
        **kwargs
    ) -> Any:
        """
        Generate LIME explanations.
        
        Args:
            inputs: Input to explain
            num_features: Number of features to include in explanation
            num_samples: Number of samples for LIME
            
        Returns:
            LIME explanation object
        """
        try:
            if self.model_type == "text" and isinstance(inputs, str):
                return self.explainer.explain_instance(
                    inputs,
                    self._model_predict_proba,
                    num_features=num_features,
                    num_samples=num_samples
                )
            
            elif self.model_type == "vision":
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.cpu().numpy()
                    if len(inputs.shape) == 4 and inputs.shape[0] == 1:  # Remove batch dim
                        inputs = inputs[0]
                    if inputs.shape[0] in [1, 3]:  # CHW -> HWC
                        inputs = inputs.transpose(1, 2, 0)
                
                return self.explainer.explain_instance(
                    inputs,
                    self._model_predict_image,
                    num_features=num_features,
                    num_samples=num_samples,
                    random_seed=42
                )
            
            else:
                print("Tabular LIME not yet implemented")
                return None
                
        except Exception as e:
            print(f"LIME explanation failed: {e}")
            return None


class CaptumExplainer(BaselineExplainer):
    """Captum-based explanations (Integrated Gradients, etc.)."""
    
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        super().__init__(model, device)
        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum is required for CaptumExplainer")
        
        self.attribution_methods = {}
        self._initialize_methods()
    
    def _initialize_methods(self):
        """Initialize attribution methods."""
        if self.model_type == "text":
            # Use layer attribution for text models
            if hasattr(self.model, 'embeddings'):
                self.attribution_methods['integrated_gradients'] = LayerIntegratedGradients(
                    self.model, self.model.embeddings.word_embeddings
                )
                self.attribution_methods['gradient_shap'] = LayerGradientShap(
                    self.model, self.model.embeddings.word_embeddings
                )
            else:
                self.attribution_methods['integrated_gradients'] = IntegratedGradients(self.model)
                self.attribution_methods['gradient_shap'] = GradientShap(self.model)
        else:
            # Standard attribution methods for vision/tabular
            self.attribution_methods['integrated_gradients'] = IntegratedGradients(self.model)
            self.attribution_methods['gradient_shap'] = GradientShap(self.model)
            self.attribution_methods['deep_lift'] = DeepLift(self.model)
    
    def explain(
        self, 
        inputs: torch.Tensor, 
        target: Optional[int] = None,
        method: str = "integrated_gradients",
        baseline: Optional[torch.Tensor] = None,
        n_steps: int = 50,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate Captum-based explanations.
        
        Args:
            inputs: Input tensor
            target: Target class index
            method: Attribution method to use
            baseline: Baseline for attribution
            n_steps: Number of integration steps
            
        Returns:
            Attribution tensor
        """
        if method not in self.attribution_methods:
            raise ValueError(f"Method {method} not available. Choose from: {list(self.attribution_methods.keys())}")
        
        explainer = self.attribution_methods[method]
        inputs = inputs.to(self.device)
        
        if baseline is None:
            baseline = torch.zeros_like(inputs)
        
        try:
            if target is None:
                # Use predicted class as target
                with torch.no_grad():
                    outputs = self.model(inputs)
                    target = torch.argmax(outputs, dim=1).item()
            
            attributions = explainer.attribute(
                inputs,
                baselines=baseline,
                target=target,
                n_steps=n_steps,
                **kwargs
            )
            
            return attributions.detach()
            
        except Exception as e:
            print(f"Captum explanation failed: {e}")
            return None


class ExplainerHub:
    """
    Unified hub for all explanation methods.
    Provides a single interface to access SHAP, LIME, Captum, and RA explanations.
    """
    
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.explainers = {}
        
        # Initialize available explainers
        if SHAP_AVAILABLE:
            try:
                self.explainers['shap'] = SHAPExplainer(model, device)
            except Exception as e:
                print(f"Failed to initialize SHAP: {e}")
        
        if LIME_AVAILABLE:
            try:
                self.explainers['lime'] = LIMEExplainer(model, device)
            except Exception as e:
                print(f"Failed to initialize LIME: {e}")
        
        if CAPTUM_AVAILABLE:
            try:
                self.explainers['captum'] = CaptumExplainer(model, device)
            except Exception as e:
                print(f"Failed to initialize Captum: {e}")
    
    def explain(
        self, 
        inputs: Any, 
        method: str = "shap", 
        **kwargs
    ) -> Any:
        """
        Generate explanations using specified method.
        
        Args:
            inputs: Input data to explain
            method: Explanation method ('shap', 'lime', 'captum')
            **kwargs: Method-specific arguments
            
        Returns:
            Method-specific explanation object
        """
        if method not in self.explainers:
            available_methods = list(self.explainers.keys())
            raise ValueError(f"Method {method} not available. Available: {available_methods}")
        
        return self.explainers[method].explain(inputs, **kwargs)
    
    def explain_text(self, text: str, method: str = "shap", **kwargs) -> Any:
        """Convenience method for text explanations."""
        return self.explain(text, method, **kwargs)
    
    def explain_image(self, image: np.ndarray, method: str = "lime", **kwargs) -> Any:
        """Convenience method for image explanations."""
        return self.explain(image, method, **kwargs)
    
    def get_available_methods(self) -> List[str]:
        """Get list of available explanation methods."""
        return list(self.explainers.keys())
    
    def compare_methods(
        self, 
        inputs: Any, 
        methods: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compare multiple explanation methods on the same input.
        
        Args:
            inputs: Input to explain
            methods: List of methods to compare (default: all available)
            **kwargs: Shared arguments for all methods
            
        Returns:
            Dictionary mapping method names to their explanations
        """
        if methods is None:
            methods = self.get_available_methods()
        
        results = {}
        
        for method in methods:
            if method in self.explainers:
                try:
                    results[method] = self.explain(inputs, method, **kwargs)
                except Exception as e:
                    results[method] = {'error': str(e)}
            else:
                results[method] = {'error': f'Method {method} not available'}
        
        return results


# Utility functions for processing explanations
def normalize_attributions(attributions: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize attribution scores.
    
    Args:
        attributions: Attribution array
        method: Normalization method ("minmax", "zscore", "sum")
        
    Returns:
        Normalized attributions
    """
    if method == "minmax":
        min_val, max_val = attributions.min(), attributions.max()
        if max_val > min_val:
            return (attributions - min_val) / (max_val - min_val)
        return attributions
    
    elif method == "zscore":
        mean_val, std_val = attributions.mean(), attributions.std()
        if std_val > 0:
            return (attributions - mean_val) / std_val
        return attributions
    
    elif method == "sum":
        total = np.abs(attributions).sum()
        if total > 0:
            return attributions / total
        return attributions
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def extract_top_features(
    attributions: np.ndarray, 
    feature_names: List[str] = None,
    k: int = 10,
    absolute: bool = True
) -> List[Tuple[str, float]]:
    """
    Extract top-k features from attributions.
    
    Args:
        attributions: Attribution scores
        feature_names: Names of features (optional)
        k: Number of top features to extract
        absolute: Whether to use absolute values for ranking
        
    Returns:
        List of (feature_name, score) tuples
    """
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(attributions))]
    
    if absolute:
        indices = np.argsort(np.abs(attributions))[-k:][::-1]
    else:
        indices = np.argsort(attributions)[-k:][::-1]
    
    return [(feature_names[i], attributions[i]) for i in indices]


if __name__ == "__main__":
    # Example usage
    import torch
    import torch.nn as nn
    
    # Simple test model
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    
    # Initialize explainer hub
    hub = ExplainerHub(model)
    
    print("Available methods:", hub.get_available_methods())
    
    # Test with random data
    test_input = torch.randn(1, 10)
    
    if 'captum' in hub.get_available_methods():
        explanation = hub.explain(test_input, method='captum', method='integrated_gradients')
        print("Captum explanation shape:", explanation.shape if explanation is not None else None)
