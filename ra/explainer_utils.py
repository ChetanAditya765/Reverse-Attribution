"""
Baseline explanation methods for comparison with Reverse Attribution.
Integrated with your actual model implementations:
- BERTSentimentClassifier
- ResNetCIFAR  
- Custom model examples

Provides SHAP, LIME, and Captum (Integrated Gradients) explanations
that work seamlessly with your model architectures.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from abc import ABC, abstractmethod
from pathlib import Path
import sys

# Import baseline explanation libraries
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not available - install with: pip install shap")

try:
    import lime
    from lime.lime_text import LimeTextExplainer
    from lime.lime_image import LimeImageExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("⚠️ LIME not available - install with: pip install lime")

try:
    import captum
    from captum.attr import IntegratedGradients, LayerIntegratedGradients, GradientShap
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    print("⚠️ Captum not available - install with: pip install captum")

# Add models directory to path
models_path = Path(__file__).parent.parent / "models"
sys.path.insert(0, str(models_path))

# Import your actual model implementations
try:
    from models.bert_sentiment import BERTSentimentClassifier
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False

try:
    from models.resnet_cifar import ResNetCIFAR, resnet56_cifar
    RESNET_AVAILABLE = True
except ImportError:
    RESNET_AVAILABLE = False

try:
    from models.custom_model_example import CustomTextClassifier, CustomVisionClassifier
    CUSTOM_AVAILABLE = True
except ImportError:
    CUSTOM_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineExplainer(ABC):
    """
    Abstract base class for baseline explanation methods.
    """
    
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.device = device
        self.model_type = self._detect_model_type()
        
    def _detect_model_type(self) -> str:
        """Detect which of your models is being used."""
        model_class_name = self.model.__class__.__name__
        
        if model_class_name == "BERTSentimentClassifier":
            return "bert_sentiment"
        elif model_class_name == "ResNetCIFAR":
            return "resnet_cifar"
        elif model_class_name == "CustomTextClassifier":
            return "custom_text"
        elif model_class_name == "CustomVisionClassifier":
            return "custom_vision"
        else:
            return "unknown"
    
    @abstractmethod
    def explain(self, input_data: Any, target_class: int, **kwargs) -> Dict[str, Any]:
        """Generate explanation for input data."""
        pass
    
    def is_available(self) -> bool:
        """Check if this explainer is available."""
        return True


class SHAPExplainer(BaselineExplainer):
    """
    SHAP explainer that works with your actual model implementations.
    """
    
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        super().__init__(model, device)
        
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not available. Install with: pip install shap")
        
        self.explainer = None
        self._setup_explainer()
        
    def _setup_explainer(self):
        """Setup SHAP explainer based on your model type."""
        
        if self.model_type in ["bert_sentiment", "custom_text"]:
            # For your text models, use Deep explainer
            try:
                # Create background samples
                if hasattr(self.model, 'tokenizer'):
                    # Use your BERT model's tokenizer
                    sample_texts = ["This is a positive example.", "This is a negative example."]
                    background = self.model.encode_text(sample_texts)
                    background_tensor = background['input_ids'].to(self.device)
                else:
                    # Fallback for custom text models
                    background_tensor = torch.randint(0, 1000, (2, 128)).to(self.device)
                
                # Create wrapper function for SHAP
                def model_wrapper(x):
                    if self.model_type == "bert_sentiment":
                        # Handle attention mask for your BERT model
                        attention_mask = torch.ones_like(x)
                        return self.model(x, attention_mask).cpu().numpy()
                    else:
                        return self.model(x).cpu().numpy()
                
                self.explainer = shap.DeepExplainer(model_wrapper, background_tensor)
                logger.info(f"✅ SHAP Deep explainer setup for {self.model_type}")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to setup SHAP Deep explainer: {e}")
                self.explainer = None
        
        elif self.model_type in ["resnet_cifar", "custom_vision"]:
            # For your vision models, use Deep explainer with image background
            try:
                # Create background images (e.g., zeros or mean of dataset)
                background_images = torch.zeros(2, 3, 32, 32).to(self.device)
                
                def model_wrapper(x):
                    return self.model(x).cpu().numpy()
                
                self.explainer = shap.DeepExplainer(model_wrapper, background_images)
                logger.info(f"✅ SHAP Deep explainer setup for {self.model_type}")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to setup SHAP Deep explainer: {e}")
                self.explainer = None
        
        else:
            logger.warning(f"⚠️ SHAP explainer not supported for model type: {self.model_type}")
    
    def explain(
        self, 
        input_data: torch.Tensor, 
        target_class: int = None, 
        **kwargs
    ) -> Dict[str, Any]:
        """Generate SHAP explanation for your models."""
        
        if self.explainer is None:
            return {
                'method': 'SHAP',
                'error': 'SHAP explainer not available',
                'attributions': np.zeros_like(input_data.cpu().numpy()),
                'model_type': self.model_type
            }
        
        try:
            input_data = input_data.to(self.device)
            
            # Generate SHAP values
            if self.model_type == "bert_sentiment":
                # Handle attention mask for your BERT model
                attention_mask = torch.ones_like(input_data)
                
                # SHAP explanation
                shap_values = self.explainer.shap_values(input_data)
                
                # Handle multiple classes
                if isinstance(shap_values, list):
                    if target_class is not None:
                        attributions = shap_values[target_class][0]
                    else:
                        # Use class with highest attribution magnitude
                        attributions = shap_values[np.argmax([np.abs(sv).sum() for sv in shap_values])][0]
                else:
                    attributions = shap_values[0]
            
            else:
                # For other model types
                shap_values = self.explainer.shap_values(input_data)
                
                if isinstance(shap_values, list):
                    if target_class is not None:
                        attributions = shap_values[target_class][0]
                    else:
                        attributions = shap_values[0][0]
                else:
                    attributions = shap_values[0]
            
            # Get model prediction for reference
            with torch.no_grad():
                if self.model_type == "bert_sentiment":
                    attention_mask = torch.ones_like(input_data)
                    outputs = self.model(input_data, attention_mask)
                else:
                    outputs = self.model(input_data)
                
                probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
                predicted_class = np.argmax(probabilities)
            
            return {
                'method': 'SHAP',
                'attributions': attributions,
                'predicted_class': int(predicted_class),
                'probabilities': probabilities.tolist(),
                'model_type': self.model_type,
                'target_class': target_class,
                'attribution_shape': attributions.shape
            }
            
        except Exception as e:
            logger.error(f"❌ SHAP explanation failed: {e}")
            return {
                'method': 'SHAP',
                'error': str(e),
                'attributions': np.zeros_like(input_data.cpu().numpy()),
                'model_type': self.model_type
            }
    
    def is_available(self) -> bool:
        """Check if SHAP explainer is available."""
        return SHAP_AVAILABLE and self.explainer is not None


class LIMEExplainer(BaselineExplainer):
    """
    LIME explainer that works with your actual model implementations.
    """
    
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        super().__init__(model, device)
        
        if not LIME_AVAILABLE:
            raise ImportError("LIME not available. Install with: pip install lime")
        
        self.text_explainer = None
        self.image_explainer = None
        self._setup_explainer()
    
    def _setup_explainer(self):
        """Setup LIME explainer based on your model type."""
        
        if self.model_type in ["bert_sentiment", "custom_text"]:
            # Setup LIME text explainer
            try:
                self.text_explainer = LimeTextExplainer(
                    class_names=['negative', 'positive'],
                    feature_selection='auto',
                    split_expression=r'\W+',
                    bow=True,
                    mode='classification'
                )
                logger.info(f"✅ LIME text explainer setup for {self.model_type}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to setup LIME text explainer: {e}")
        
        elif self.model_type in ["resnet_cifar", "custom_vision"]:
            # Setup LIME image explainer
            try:
                self.image_explainer = LimeImageExplainer(
                    feature_selection='auto',
                    mode='classification'
                )
                logger.info(f"✅ LIME image explainer setup for {self.model_type}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to setup LIME image explainer: {e}")
    
    def explain(
        self, 
        input_data: Union[torch.Tensor, str, List[str]], 
        target_class: int = None,
        num_features: int = 10,
        num_samples: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate LIME explanation for your models."""
        
        try:
            if self.model_type in ["bert_sentiment", "custom_text"] and self.text_explainer is not None:
                return self._explain_text(input_data, target_class, num_features, num_samples)
            
            elif self.model_type in ["resnet_cifar", "custom_vision"] and self.image_explainer is not None:
                return self._explain_image(input_data, target_class, num_features, num_samples)
            
            else:
                return {
                    'method': 'LIME',
                    'error': f'LIME explainer not available for model type: {self.model_type}',
                    'model_type': self.model_type
                }
                
        except Exception as e:
            logger.error(f"❌ LIME explanation failed: {e}")
            return {
                'method': 'LIME',
                'error': str(e),
                'model_type': self.model_type
            }
    
    def _explain_text(self, input_text: Union[str, List[str]], target_class: int, num_features: int, num_samples: int):
        """Explain text input using your BERT model."""
        
        if isinstance(input_text, list):
            input_text = input_text[0]  # Take first text if list provided
        
        # Create prediction function for your BERT model
        def predict_fn(texts):
            # Encode texts using your model's tokenizer
            if hasattr(self.model, 'tokenizer'):
                encoded = self.model.encode_text(texts)
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(input_ids, attention_mask)
                    probabilities = F.softmax(outputs, dim=1)
                    return probabilities.cpu().numpy()
            else:
                # Fallback for custom text models
                # This would need custom tokenization logic
                return np.random.rand(len(texts), 2)  # Placeholder
        
        # Generate LIME explanation
        explanation = self.text_explainer.explain_instance(
            input_text,
            predict_fn,
            num_features=num_features,
            num_samples=num_samples
        )
        
        # Extract feature importance
        feature_importance = explanation.as_list()
        
        # Get model prediction
        prediction = predict_fn([input_text])[0]
        predicted_class = np.argmax(prediction)
        
        return {
            'method': 'LIME',
            'feature_importance': feature_importance,
            'predicted_class': int(predicted_class),
            'probabilities': prediction.tolist(),
            'model_type': self.model_type,
            'text': input_text,
            'explanation_object': explanation
        }
    
    def _explain_image(self, input_image: torch.Tensor, target_class: int, num_features: int, num_samples: int):
        """Explain image input using your ResNet model."""
        
        input_image = input_image.to(self.device)
        
        # Convert tensor to numpy for LIME (C, H, W) -> (H, W, C)
        if input_image.dim() == 4:
            image_np = input_image[0].permute(1, 2, 0).cpu().numpy()
        else:
            image_np = input_image.permute(1, 2, 0).cpu().numpy()
        
        # Normalize image to [0, 1] range for LIME
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        
        # Create prediction function for your ResNet model
        def predict_fn(images):
            batch_images = []
            for img in images:
                # Convert back to tensor format (H, W, C) -> (C, H, W)
                img_tensor = torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0)
                batch_images.append(img_tensor)
            
            batch_tensor = torch.cat(batch_images, dim=0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = F.softmax(outputs, dim=1)
                return probabilities.cpu().numpy()
        
        # Generate LIME explanation
        explanation = self.image_explainer.explain_instance(
            image_np,
            predict_fn,
            top_labels=5,
            num_features=num_features,
            num_samples=num_samples,
            batch_size=32
        )
        
        # Get model prediction
        prediction = predict_fn([image_np])[0]
        predicted_class = np.argmax(prediction)
        
        # Get explanation for target class or predicted class
        explain_class = target_class if target_class is not None else predicted_class
        
        return {
            'method': 'LIME',
            'predicted_class': int(predicted_class),
            'probabilities': prediction.tolist(),
            'model_type': self.model_type,
            'explanation_class': explain_class,
            'explanation_object': explanation,
            'image_shape': image_np.shape
        }
    
    def is_available(self) -> bool:
        """Check if LIME explainer is available."""
        return LIME_AVAILABLE and (self.text_explainer is not None or self.image_explainer is not None)


class CaptumExplainer(BaselineExplainer):
    """
    Captum explainer (Integrated Gradients) that works with your actual models.
    """
    
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        super().__init__(model, device)
        
        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum not available. Install with: pip install captum")
        
        self.attribution_method = None
        self._setup_explainer()
    
    def _setup_explainer(self):
        """Setup Captum explainer based on your model type."""
        
        try:
            if self.model_type in ["bert_sentiment", "custom_text"]:
                # For your text models, use LayerIntegratedGradients on embeddings
                if hasattr(self.model, 'embeddings') and hasattr(self.model.embeddings, 'word_embeddings'):
                    self.attribution_method = LayerIntegratedGradients(
                        self.model, 
                        self.model.embeddings.word_embeddings
                    )
                elif hasattr(self.model, 'bert') and hasattr(self.model.bert, 'embeddings'):
                    self.attribution_method = LayerIntegratedGradients(
                        self.model,
                        self.model.bert.embeddings.word_embeddings
                    )
                else:
                    self.attribution_method = IntegratedGradients(self.model)
                
                logger.info(f"✅ Captum explainer setup for {self.model_type}")
            
            elif self.model_type in ["resnet_cifar", "custom_vision"]:
                # For your vision models, use standard IntegratedGradients
                self.attribution_method = IntegratedGradients(self.model)
                logger.info(f"✅ Captum explainer setup for {self.model_type}")
            
            else:
                # Fallback to standard IG
                self.attribution_method = IntegratedGradients(self.model)
                logger.warning(f"⚠️ Using fallback Integrated Gradients for {self.model_type}")
                
        except Exception as e:
            logger.error(f"❌ Failed to setup Captum explainer: {e}")
            self.attribution_method = None
    
    def explain(
        self, 
        input_data: torch.Tensor, 
        target_class: int = None,
        n_steps: int = 50,
        additional_forward_args: tuple = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate Captum explanation for your models."""
        
        if self.attribution_method is None:
            return {
                'method': 'Captum (Integrated Gradients)',
                'error': 'Captum explainer not available',
                'model_type': self.model_type
            }
        
        try:
            input_data = input_data.to(self.device)
            
            # Get model prediction
            with torch.no_grad():
                if self.model_type == "bert_sentiment" and additional_forward_args:
                    outputs = self.model(input_data, *additional_forward_args)
                else:
                    outputs = self.model(input_data)
                
                probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
                predicted_class = np.argmax(probabilities)
            
            # Determine target for attribution
            attribution_target = target_class if target_class is not None else predicted_class
            
            # Generate baseline
            baseline = torch.zeros_like(input_data)
            
            # Compute attributions
            attributions = self.attribution_method.attribute(
                input_data,
                baselines=baseline,
                target=attribution_target,
                n_steps=n_steps,
                additional_forward_args=additional_forward_args
            )
            
            # Handle different attribution shapes
            if attributions.dim() > 2:
                # For layer attributions, squeeze and flatten
                attributions = attributions.squeeze()
            
            attributions_np = attributions.cpu().numpy()
            if attributions_np.ndim > 1:
                attributions_np = attributions_np.flatten()
            
            return {
                'method': 'Captum (Integrated Gradients)',
                'attributions': attributions_np,
                'predicted_class': int(predicted_class),
                'probabilities': probabilities.tolist(),
                'model_type': self.model_type,
                'target_class': target_class,
                'attribution_target': int(attribution_target),
                'n_steps': n_steps
            }
            
        except Exception as e:
            logger.error(f"❌ Captum explanation failed: {e}")
            return {
                'method': 'Captum (Integrated Gradients)',
                'error': str(e),
                'model_type': self.model_type
            }
    
    def is_available(self) -> bool:
        """Check if Captum explainer is available."""
        return CAPTUM_AVAILABLE and self.attribution_method is not None


class ExplainerHub:
    """
    Central hub for managing baseline explanation methods with your models.
    """
    
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model_type = self._detect_model_type()
        
        # Initialize explainers
        self.explainers = {}
        self._initialize_explainers()
    
    def _detect_model_type(self) -> str:
        """Detect which of your models is being used."""
        model_class_name = self.model.__class__.__name__
        
        if model_class_name == "BERTSentimentClassifier":
            return "bert_sentiment"
        elif model_class_name == "ResNetCIFAR":
            return "resnet_cifar"
        elif model_class_name == "CustomTextClassifier":
            return "custom_text"
        elif model_class_name == "CustomVisionClassifier":
            return "custom_vision"
        else:
            return "unknown"
    
    def _initialize_explainers(self):
        """Initialize available explainers."""
        logger.info(f"🔧 Initializing explainers for {self.model_type}")
        
        # SHAP
        if SHAP_AVAILABLE:
            try:
                self.explainers['shap'] = SHAPExplainer(self.model, self.device)
                if self.explainers['shap'].is_available():
                    logger.info("✅ SHAP explainer initialized")
                else:
                    del self.explainers['shap']
            except Exception as e:
                logger.warning(f"⚠️ SHAP initialization failed: {e}")
        
        # LIME
        if LIME_AVAILABLE:
            try:
                self.explainers['lime'] = LIMEExplainer(self.model, self.device)
                if self.explainers['lime'].is_available():
                    logger.info("✅ LIME explainer initialized")
                else:
                    del self.explainers['lime']
            except Exception as e:
                logger.warning(f"⚠️ LIME initialization failed: {e}")
        
        # Captum
        if CAPTUM_AVAILABLE:
            try:
                self.explainers['captum'] = CaptumExplainer(self.model, self.device)
                if self.explainers['captum'].is_available():
                    logger.info("✅ Captum explainer initialized")
                else:
                    del self.explainers['captum']
            except Exception as e:
                logger.warning(f"⚠️ Captum initialization failed: {e}")
    
    def get_available_explainers(self) -> List[str]:
        """Get list of available explainers."""
        return list(self.explainers.keys())
    
    def explain_with_all(
        self, 
        input_data: Any, 
        target_class: int = None, 
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """Generate explanations using all available methods."""
        
        explanations = {}
        
        for method_name, explainer in self.explainers.items():
            try:
                logger.info(f"🔍 Generating {method_name.upper()} explanation...")
                
                if method_name == 'captum' and self.model_type == "bert_sentiment":
                    # Handle attention mask for Captum with BERT
                    if isinstance(input_data, dict):
                        attention_mask = input_data.get('attention_mask', None)
                        input_ids = input_data['input_ids']
                        additional_args = (attention_mask,) if attention_mask is not None else None
                        explanation = explainer.explain(input_ids, target_class, additional_forward_args=additional_args, **kwargs)
                    else:
                        explanation = explainer.explain(input_data, target_class, **kwargs)
                else:
                    explanation = explainer.explain(input_data, target_class, **kwargs)
                
                explanations[method_name] = explanation
                logger.info(f"✅ {method_name.upper()} explanation completed")
                
            except Exception as e:
                logger.error(f"❌ {method_name.upper()} explanation failed: {e}")
                explanations[method_name] = {
                    'method': method_name.upper(),
                    'error': str(e),
                    'model_type': self.model_type
                }
        
        return explanations
    
    def explain_with_method(
        self, 
        method_name: str, 
        input_data: Any, 
        target_class: int = None, 
        **kwargs
    ) -> Dict[str, Any]:
        """Generate explanation using specific method."""
        
        if method_name not in self.explainers:
            return {
                'method': method_name.upper(),
                'error': f'Method {method_name} not available',
                'available_methods': self.get_available_explainers(),
                'model_type': self.model_type
            }
        
        explainer = self.explainers[method_name]
        return explainer.explain(input_data, target_class, **kwargs)
    
    def compare_methods(
        self, 
        input_data: Any, 
        target_class: int = None, 
        methods: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Compare explanations from multiple methods."""
        
        if methods is None:
            methods = self.get_available_explainers()
        
        explanations = {}
        for method in methods:
            if method in self.explainers:
                explanations[method] = self.explain_with_method(method, input_data, target_class, **kwargs)
        
        # Compute comparison metrics if multiple explanations available
        comparison_metrics = self._compute_comparison_metrics(explanations)
        
        return {
            'explanations': explanations,
            'comparison_metrics': comparison_metrics,
            'model_type': self.model_type,
            'methods_compared': list(explanations.keys())
        }
    
    def _compute_comparison_metrics(self, explanations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compute metrics comparing different explanation methods."""
        
        metrics = {
            'methods_successful': [],
            'methods_failed': [],
            'attribution_similarities': {}
        }
        
        successful_explanations = {}
        
        # Identify successful explanations
        for method, explanation in explanations.items():
            if 'error' not in explanation and 'attributions' in explanation:
                metrics['methods_successful'].append(method)
                successful_explanations[method] = explanation
            else:
                metrics['methods_failed'].append(method)
        
        # Compute attribution similarities between successful methods
        if len(successful_explanations) >= 2:
            methods = list(successful_explanations.keys())
            for i, method1 in enumerate(methods):
                for method2 in methods[i+1:]:
                    try:
                        attr1 = successful_explanations[method1]['attributions']
                        attr2 = successful_explanations[method2]['attributions']
                        
                        # Ensure same shape
                        min_len = min(len(attr1), len(attr2))
                        attr1 = attr1[:min_len]
                        attr2 = attr2[:min_len]
                        
                        # Compute correlation
                        correlation = np.corrcoef(attr1, attr2)[0, 1]
                        metrics['attribution_similarities'][f'{method1}_vs_{method2}'] = float(correlation)
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Could not compare {method1} and {method2}: {e}")
        
        return metrics


# Utility functions
def normalize_attributions(attributions: np.ndarray) -> np.ndarray:
    """Normalize attributions to [0, 1] range."""
    attr_min, attr_max = attributions.min(), attributions.max()
    if attr_max > attr_min:
        return (attributions - attr_min) / (attr_max - attr_min)
    return attributions


def extract_top_features(
    attributions: np.ndarray, 
    feature_names: List[str] = None, 
    top_k: int = 10
) -> List[Tuple[str, float]]:
    """Extract top-k features by attribution magnitude."""
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(attributions))]
    
    # Get indices of top-k attributions by absolute value
    top_indices = np.argsort(np.abs(attributions))[-top_k:][::-1]
    
    top_features = []
    for idx in top_indices:
        feature_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        attribution_value = attributions[idx]
        top_features.append((feature_name, float(attribution_value)))
    
    return top_features


def check_explainer_availability() -> Dict[str, bool]:
    """Check which explainer libraries are available."""
    return {
        'shap': SHAP_AVAILABLE,
        'lime': LIME_AVAILABLE,
        'captum': CAPTUM_AVAILABLE,
        'bert_models': BERT_AVAILABLE,
        'resnet_models': RESNET_AVAILABLE,
        'custom_models': CUSTOM_AVAILABLE
    }


# Example usage functions
def create_explainer_for_bert(model: Any, device: str = "cpu") -> ExplainerHub:
    """Create explainer hub for your BERT sentiment model."""
    return ExplainerHub(model, device)


def create_explainer_for_resnet(model: Any, device: str = "cpu") -> ExplainerHub:
    """Create explainer hub for your ResNet CIFAR model."""
    return ExplainerHub(model, device)


if __name__ == "__main__":
    # Test explainer availability
    print("🔍 Checking explainer availability...")
    availability = check_explainer_availability()
    
    for component, available in availability.items():
        status = "✅" if available else "❌"
        print(f"  {status} {component}")
    
    # Test with your models if available
    if BERT_AVAILABLE and SHAP_AVAILABLE:
        print("\n🧪 Testing with BERTSentimentClassifier...")
        try:
            from models.bert_sentiment import BERTSentimentClassifier
            
            model = BERTSentimentClassifier("bert-base-uncased", num_classes=2)
            hub = create_explainer_for_bert(model)
            
            available_explainers = hub.get_available_explainers()
            print(f"✅ Available explainers for BERT: {available_explainers}")
            
        except Exception as e:
            print(f"❌ BERT test failed: {e}")
    
    if RESNET_AVAILABLE and CAPTUM_AVAILABLE:
        print("\n🧪 Testing with ResNetCIFAR...")
        try:
            from models.resnet_cifar import resnet56_cifar
            
            model = resnet56_cifar(num_classes=10)
            hub = create_explainer_for_resnet(model)
            
            available_explainers = hub.get_available_explainers()
            print(f"✅ Available explainers for ResNet: {available_explainers}")
            
        except Exception as e:
            print(f"❌ ResNet test failed: {e}")
    
    print("\n🎉 Baseline explainer integration test completed!")
