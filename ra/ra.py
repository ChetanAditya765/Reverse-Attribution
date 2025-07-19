"""
Reverse Attribution (RA) - Core Implementation
Based on: "Reverse Attribution: Explaining Model Uncertainty and Failures via Counter-Evidence Analysis"
Author: Chetan Aditya Lakka
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from captum.attr import IntegratedGradients, LayerIntegratedGradients


class ReverseAttribution:
    """
    Core Reverse Attribution implementation following Algorithm 1 from the paper.
    
    This class identifies counter-evidence features that suppress the true class
    prediction through signed attribution analysis and counterfactual masking.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        baseline: Optional[Union[torch.Tensor, str]] = None,
        device: str = None
    ):
        """
        Initialize Reverse Attribution explainer.
        
        Args:
            model: PyTorch model to explain
            baseline: Baseline input for attribution computation
            device: Device to run computations on
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.baseline = baseline
        
        # Select appropriate attribution method based on model type
        if hasattr(model, 'embeddings') and hasattr(model.embeddings, 'word_embeddings'):
            # Text models (BERT, RoBERTa, etc.)
            self.attribution_method = LayerIntegratedGradients(
                self.model, 
                self.model.embeddings.word_embeddings
            )
        else:
            # Vision models or other architectures
            self.attribution_method = IntegratedGradients(self.model)
    
    def explain(
        self,
        x: torch.Tensor,
        y_true: int,
        top_m: int = 5,
        n_steps: int = 50
    ) -> Dict:
        """
        Generate reverse attribution explanation for input x.
        
        Args:
            x: Input tensor to explain
            y_true: True label for the input
            top_m: Number of top counter-evidence features to return
            n_steps: Number of integration steps for attribution
            
        Returns:
            Dictionary containing:
                - counter_evidence: List of (feature_idx, attribution, delta) tuples
                - a_flip: Attribution-flip instability score
                - phi: Complete attribution vector
                - y_hat: Predicted class
                - runner_up: Second most likely class
        """
        x = x.to(self.device)
        
        with torch.no_grad():
            # Forward pass to get predictions
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)
            y_hat = logits.argmax(dim=-1).item()
            
            # Get runner-up class (second highest probability)
            top2_probs, top2_classes = torch.topk(probs, 2, dim=-1)
            runner_up = top2_classes[0, 1].item()
        
        # Step 1: Compute signed attributions w.r.t. predicted class
        phi = self._compute_attributions(x, target=y_hat, n_steps=n_steps)
        
        # Step 2: Filter negative attributions (counter-evidence)
        negative_mask = phi < 0
        negative_indices = torch.nonzero(negative_mask, as_tuple=True)[0]
        
        if len(negative_indices) == 0:
            return {
                "counter_evidence": [],
                "a_flip": 0.0,
                "phi": phi.cpu().numpy(),
                "y_hat": y_hat,
                "runner_up": runner_up
            }
        
        # Step 3: Counterfactual pull - quantify suppression effect
        delta_values = []
        original_prob = probs[0, y_true].item()
        
        for idx in negative_indices:
            # Create masked input
            x_masked = self._mask_feature(x, idx.item())
            
            with torch.no_grad():
                masked_logits = self.model(x_masked)
                masked_probs = F.softmax(masked_logits, dim=-1)
                masked_prob = masked_probs[0, y_true].item()
            
            # Calculate counterfactual delta
            delta = masked_prob - original_prob
            delta_values.append(delta)
        
        # Rank by absolute suppression strength
        counter_evidence = []
        for i, (idx, delta) in enumerate(zip(negative_indices, delta_values)):
            counter_evidence.append((
                idx.item(),
                phi[idx].item(),
                delta
            ))
        
        # Sort by absolute delta (suppression strength) and take top-m
        counter_evidence.sort(key=lambda x: abs(x[2]), reverse=True)
        counter_evidence = counter_evidence[:top_m]
        
        # Step 4: Attribution-Flip Score
        phi_runner = self._compute_attributions(x, target=runner_up, n_steps=n_steps)
        a_flip = 0.5 * torch.sum(torch.abs(phi - phi_runner)).item()
        
        return {
            "counter_evidence": counter_evidence,
            "a_flip": a_flip,
            "phi": phi.cpu().numpy(),
            "y_hat": y_hat,
            "runner_up": runner_up
        }
    
    def _compute_attributions(
        self, 
        x: torch.Tensor, 
        target: int, 
        n_steps: int = 50
    ) -> torch.Tensor:
        """Compute attribution scores for given target class."""
        attributions = self.attribution_method.attribute(
            x,
            target=target,
            n_steps=n_steps,
            baselines=self._get_baseline(x)
        )
        
        # Flatten to 1D for easier processing
        return attributions.squeeze().flatten()
    
    def _mask_feature(self, x: torch.Tensor, feature_idx: int) -> torch.Tensor:
        """Create masked version of input by replacing feature at index with baseline."""
        x_masked = x.clone()
        baseline = self._get_baseline(x)
        
        if baseline is not None:
            x_masked_flat = x_masked.flatten()
            baseline_flat = baseline.flatten()
            x_masked_flat[feature_idx] = baseline_flat[feature_idx]
            x_masked = x_masked_flat.reshape(x.shape)
        else:
            # Use zero baseline if none provided
            x_masked.flatten()[feature_idx] = 0.0
            
        return x_masked
    
    def _get_baseline(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Get baseline tensor for attribution computation."""
        if self.baseline is None:
            return torch.zeros_like(x)
        elif isinstance(self.baseline, torch.Tensor):
            return self.baseline.to(x.device)
        else:
            # For text models, might use special tokens
            return torch.zeros_like(x)
