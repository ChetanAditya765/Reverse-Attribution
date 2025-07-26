"""
Updated Reverse Attribution implementation that properly integrates with your models.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from captum.attr import IntegratedGradients, LayerIntegratedGradients
import sys
from pathlib import Path

# Import your models for proper integration
models_path = Path(__file__).parent.parent / "models"
sys.path.insert(0, str(models_path))


class ReverseAttribution:
    """
    RA implementation that properly detects and works with your specific models.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        baseline: Optional[Union[torch.Tensor, str]] = None,
        device: str = None
    ):
        """
        Initialize RA explainer with proper model detection.
        """
        from ra.device_utils import device as auto_device
        self.device = device or auto_device
        self.model = model.to(self.device).eval()

        self.baseline = baseline
        self.model_type = self._detect_model_type()
        
        # Initialize attribution method based on your actual model type
        self.attribution_method = self._setup_attribution_method()
    
    def _detect_model_type(self) -> str:
        """
        Detect which of your models is being used.
        """
        model_class_name = self.model.__class__.__name__
        
        # Check for your specific model implementations
        if model_class_name == "BERTSentimentClassifier":
            return "bert_sentiment"
        elif model_class_name == "ResNetCIFAR":
            return "resnet_cifar"
        elif model_class_name == "CustomTextClassifier":
            return "custom_text"
        elif model_class_name == "CustomVisionClassifier":
            return "custom_vision"
        elif hasattr(self.model, 'embeddings') and hasattr(self.model.embeddings, 'word_embeddings'):
            return "text_transformer"
        elif any(isinstance(module, torch.nn.Conv2d) for module in self.model.modules()):
            return "vision_cnn"
        else:
            return "unknown"
    
    def _setup_attribution_method(self):
        """
        Setup attribution method based on your specific model architecture.
        """
        if self.model_type in ["bert_sentiment", "custom_text", "text_transformer"]:
            # For your text models, use layer attribution on embeddings
            if hasattr(self.model, 'embeddings') and hasattr(self.model.embeddings, 'word_embeddings'):
                return LayerIntegratedGradients(self.model, self.model.embeddings.word_embeddings)
            elif hasattr(self.model, 'bert') and hasattr(self.model.bert, 'embeddings'):
                return LayerIntegratedGradients(self.model, self.model.bert.embeddings.word_embeddings)
            else:
                return IntegratedGradients(self.model)
        
        elif self.model_type in ["resnet_cifar", "custom_vision", "vision_cnn"]:
            # For your vision models, use standard integrated gradients
            return IntegratedGradients(self.model)
        
        else:
            # Fallback to standard IG
            return IntegratedGradients(self.model)
    
    def explain(
        self,
        x: torch.Tensor,
        y_true: int,
        top_m: int = 5,
        n_steps: int = 50,
        additional_forward_args: tuple = None
    ) -> Dict:
        """
        Generate RA explanation that works with your specific models.
        """
        x = x.to(self.device)
        
        with torch.no_grad():
            # Handle different forward pass signatures for your models
            if self.model_type == "bert_sentiment":
                # Your BERT model might need attention_mask
                if additional_forward_args:
                    logits = self.model(x, *additional_forward_args)
                else:
                    logits = self.model(x)
            else:
                # Standard forward pass for other models
                logits = self.model(x)
            
            probs = F.softmax(logits, dim=-1)
            y_hat = logits.argmax(dim=-1).item()
            
            # Get runner-up class
            top2_probs, top2_classes = torch.topk(probs, 2, dim=-1)
            runner_up = top2_classes[0, 1].item()
        
        # Compute attributions using the appropriate method
        phi = self._compute_attributions(x, target=y_hat, n_steps=n_steps, additional_forward_args=additional_forward_args)
        
        # Filter negative attributions (counter-evidence)
        negative_mask = phi < 0
        negative_indices = torch.nonzero(negative_mask, as_tuple=True)[0]
        
        if len(negative_indices) == 0:
            return {
                "counter_evidence": [],
                "a_flip": 0.0,
                "phi": phi.cpu().numpy(),
                "y_hat": y_hat,
                "runner_up": runner_up,
                "model_type": self.model_type
            }
        
        # Counterfactual analysis
        delta_values = []
        original_prob = probs[0, y_true].item()
        
        for idx in negative_indices[:20]:  # Limit for efficiency
            x_masked = self._mask_feature(x, idx.item())
            
            with torch.no_grad():
                if self.model_type == "bert_sentiment" and additional_forward_args:
                    masked_logits = self.model(x_masked, *additional_forward_args)
                else:
                    masked_logits = self.model(x_masked)
                    
                masked_probs = F.softmax(masked_logits, dim=-1)
                masked_prob = masked_probs[0, y_true].item()
            
            delta = masked_prob - original_prob
            delta_values.append(delta)
        
        # Build counter-evidence list
        counter_evidence = []
        for i, (idx, delta) in enumerate(zip(negative_indices, delta_values)):
            counter_evidence.append((
                idx.item(),
                phi[idx].item(),
                delta
            ))
        
        # Sort by suppression strength
        counter_evidence.sort(key=lambda x: abs(x[2]), reverse=True)
        counter_evidence = counter_evidence[:top_m]
        
        # Compute A-Flip score
        phi_runner = self._compute_attributions(x, target=runner_up, n_steps=n_steps, additional_forward_args=additional_forward_args)
        a_flip = 0.5 * torch.sum(torch.abs(phi - phi_runner)).item()
        
        return {
            "counter_evidence": counter_evidence,
            "a_flip": a_flip,
            "phi": phi.cpu().numpy(),
            "y_hat": y_hat,
            "runner_up": runner_up,
            "model_type": self.model_type
        }
    
    def _compute_attributions(
        self, 
        x: torch.Tensor, 
        target: int, 
        n_steps: int = 50,
        additional_forward_args: tuple = None
    ) -> torch.Tensor:
        """Compute attributions with proper handling of your model types."""
        
        try:
            attributions = self.attribution_method.attribute(
                x,
                target=target,
                n_steps=n_steps,
                baselines=self._get_baseline(x),
                additional_forward_args=additional_forward_args
            )
            
            # Handle different attribution shapes
            if attributions.dim() > 2:
                # For layer attributions or multi-dimensional outputs
                attributions = attributions.squeeze()
            
            return attributions.flatten()
            
        except Exception as e:
            print(f"Attribution computation failed: {e}")
            # Fallback to gradient-based attribution
            x.requires_grad_(True)
            
            if additional_forward_args:
                logits = self.model(x, *additional_forward_args)
            else:
                logits = self.model(x)
                
            loss = logits[0, target]
            loss.backward()
            
            return x.grad.flatten()
    
    def _mask_feature(self, x: torch.Tensor, feature_idx: int) -> torch.Tensor:
        """Create masked input for counterfactual analysis."""
        x_masked = x.clone()
        
        if self.model_type in ["bert_sentiment", "custom_text"]:
            if hasattr(self.model, 'tokenizer'):
                mask_token_id = self.model.tokenizer.mask_token_id
                x_masked_flat = x_masked.view(-1)
                if feature_idx < x_masked_flat.size(0):
                    x_masked_flat[feature_idx] = mask_token_id
                x_masked = x_masked_flat.view(x.shape)
            else:
                baseline = self._get_baseline(x)
                if baseline is not None:
                    x_masked_flat = x_masked.view(-1)
                    baseline_flat = baseline.view(-1)
                    if feature_idx < x_masked_flat.size(0):
                        x_masked_flat[feature_idx] = baseline_flat[feature_idx]
                    x_masked = x_masked_flat.view(x.shape)
        else:
            baseline = self._get_baseline(x)
            if baseline is not None:
                x_masked_flat = x_masked.view(-1)
                baseline_flat = baseline.view(-1)
                if feature_idx < x_masked_flat.size(0):
                    x_masked_flat[feature_idx] = baseline_flat[feature_idx]
                x_masked = x_masked_flat.view(x.shape)
            else:
                x_masked_flat = x_masked.view(-1)
                if feature_idx < x_masked_flat.size(0):
                    x_masked_flat[feature_idx] = 0.0
                x_masked = x_masked_flat.view(x.shape)
        
        return x_masked

    def _get_baseline(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Get appropriate baseline for your model type."""
        if self.model_type in ["bert_sentiment", "custom_text"]:
            if hasattr(self.model, 'tokenizer'):
                mask_token_id = self.model.tokenizer.mask_token_id
                return torch.full_like(x, mask_token_id)
            else:
                return torch.zeros_like(x)
        elif self.model_type in ["resnet_cifar", "custom_vision"]:
            # Use CIFAR-10 mean for vision models
            mean_values = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
            return mean_values.expand_as(x).to(x.device)
        else:
            return torch.zeros_like(x)

