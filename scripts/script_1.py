# 1. Core RA Algorithm Implementation
core_ra_code = '''"""
Reverse Attribution (RA) Core Implementation
Based on "Reverse Attribution: Explaining Model Uncertainty and Failures via Counter-Evidence Analysis"

This module implements the complete 4-step RA algorithm:
1. Compute signed attributions
2. Filter negative attributions  
3. Counterfactual pull calculation
4. Attribution-Flip (A-Flip) score computation
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
import logging

@dataclass
class RAResult:
    """Result container for Reverse Attribution analysis."""
    counter_evidence: List[Tuple[int, float, float]]  # (feature_idx, attribution, pull)
    a_flip_score: float
    negative_attributions: np.ndarray
    counterfactual_pulls: np.ndarray
    predicted_class: int
    true_class: int
    runner_up_class: int
    confidence_original: float
    confidence_after_masking: Optional[float] = None
    
    def get_top_counter_evidence(self, k: int = 5) -> List[Tuple[int, float, float]]:
        """Get top-k counter evidence features ranked by counterfactual pull."""
        return sorted(self.counter_evidence, key=lambda x: x[2], reverse=True)[:k]

class ReverseAttribution:
    """
    Reverse Attribution framework for identifying counter-evidence features.
    
    Implements the complete RA algorithm from the JMLR paper, providing model-agnostic
    explanations that surface features suppressing the true class.
    """
    
    def __init__(self, 
                 attribution_method: str = 'shap',
                 top_m: int = 5,
                 baseline_strategy: str = 'zero',
                 epsilon: float = 1e-6):
        """
        Initialize Reverse Attribution framework.
        
        Args:
            attribution_method: Attribution method ('shap', 'ig', 'lime')
            top_m: Number of top counter-evidence features to return
            baseline_strategy: Baseline computation strategy ('zero', 'mean', 'blur')
            epsilon: Small constant for numerical stability
        """
        self.attribution_method = attribution_method
        self.top_m = top_m
        self.baseline_strategy = baseline_strategy
        self.epsilon = epsilon
        self.logger = logging.getLogger(__name__)
        
    def compute_reverse_attribution(self,
                                  model_wrapper,
                                  x: Union[np.ndarray, torch.Tensor],
                                  y_true: int,
                                  baseline: Optional[Union[np.ndarray, torch.Tensor]] = None) -> RAResult:
        """
        Main RA computation following the 4-step algorithm from the paper.
        
        Args:
            model_wrapper: Wrapped model with predict_proba method
            x: Input instance
            y_true: Ground truth label
            baseline: Baseline input (computed if None)
            
        Returns:
            RAResult containing counter-evidence analysis
        """
        self.logger.info(f"Computing Reverse Attribution for sample with true label {y_true}")
        
        # Ensure inputs are numpy arrays
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if baseline is not None and isinstance(baseline, torch.Tensor):
            baseline = baseline.detach().cpu().numpy()
            
        # Get model prediction
        probs = model_wrapper.predict_proba(x.reshape(1, -1))[0]
        y_pred = np.argmax(probs)
        confidence_original = probs[y_pred]
        
        # Get runner-up class (second highest probability)
        sorted_indices = np.argsort(probs)
        y_runner_up = sorted_indices[-2] if sorted_indices[-1] == y_pred else sorted_indices[-1]
        
        self.logger.info(f"Predicted: {y_pred} (conf: {confidence_original:.3f}), True: {y_true}, Runner-up: {y_runner_up}")
        
        # Compute baseline if not provided
        if baseline is None:
            baseline = self._compute_baseline(x, model_wrapper)
            
        # Step 1: Compute signed attributions
        attributions_pred = self._compute_attributions(model_wrapper, x, y_pred, baseline)
        attributions_true = self._compute_attributions(model_wrapper, x, y_true, baseline)
        attributions_runner_up = self._compute_attributions(model_wrapper, x, y_runner_up, baseline)
        
        # Step 2: Filter negative attributions (counter-evidence)
        negative_mask = attributions_pred < 0
        negative_indices = np.where(negative_mask)[0]
        negative_attributions = attributions_pred[negative_mask]
        
        self.logger.info(f"Found {len(negative_indices)} negative attribution features")
        
        # Step 3: Counterfactual pull calculation
        counterfactual_pulls = self._compute_counterfactual_pulls(
            model_wrapper, x, y_true, negative_indices, baseline
        )
        
        # Step 4: Attribution-Flip (A-Flip) score
        a_flip_score = self._compute_a_flip_score(attributions_pred, attributions_runner_up)
        
        # Rank counter-evidence by counterfactual pull
        counter_evidence = []
        for idx, neg_idx in enumerate(negative_indices):
            counter_evidence.append((
                neg_idx,
                float(negative_attributions[idx]),
                float(counterfactual_pulls[idx])
            ))
        
        # Sort by counterfactual pull (descending)
        counter_evidence.sort(key=lambda x: x[2], reverse=True)
        
        # Take top-m features
        top_counter_evidence = counter_evidence[:self.top_m]
        
        self.logger.info(f"A-Flip score: {a_flip_score:.4f}")
        self.logger.info(f"Top counter-evidence features: {[idx for idx, _, _ in top_counter_evidence]}")
        
        return RAResult(
            counter_evidence=top_counter_evidence,
            a_flip_score=a_flip_score,
            negative_attributions=attributions_pred[negative_mask],
            counterfactual_pulls=counterfactual_pulls,
            predicted_class=y_pred,
            true_class=y_true,
            runner_up_class=y_runner_up,
            confidence_original=confidence_original
        )
    
    def _compute_baseline(self, x: np.ndarray, model_wrapper) -> np.ndarray:
        """Compute baseline input based on strategy."""
        if self.baseline_strategy == 'zero':
            return np.zeros_like(x)
        elif self.baseline_strategy == 'mean':
            # Approximate with global mean - in practice, use dataset statistics
            return np.mean(x) * np.ones_like(x)
        elif self.baseline_strategy == 'blur':
            # For images, use blurred version; for text, use padding tokens
            if len(x.shape) > 1:  # Assume image
                from scipy.ndimage import gaussian_filter
                return gaussian_filter(x, sigma=3)
            else:  # Assume tabular/text
                return np.zeros_like(x)
        else:
            return np.zeros_like(x)
    
    def _compute_attributions(self, model_wrapper, x: np.ndarray, target_class: int, 
                            baseline: np.ndarray) -> np.ndarray:
        """Compute signed attributions for target class."""
        if self.attribution_method == 'shap':
            return self._compute_shap_attributions(model_wrapper, x, target_class, baseline)
        elif self.attribution_method == 'ig':
            return self._compute_ig_attributions(model_wrapper, x, target_class, baseline)
        elif self.attribution_method == 'lime':
            return self._compute_lime_attributions(model_wrapper, x, target_class)
        else:
            raise ValueError(f"Unknown attribution method: {self.attribution_method}")
    
    def _compute_shap_attributions(self, model_wrapper, x: np.ndarray, target_class: int,
                                 baseline: np.ndarray) -> np.ndarray:
        """Compute SHAP attributions using sampling approximation."""
        try:
            import shap
            
            # Create a wrapper function for SHAP that returns target class probability
            def f(X):
                if len(X.shape) == 1:
                    X = X.reshape(1, -1)
                probs = model_wrapper.predict_proba(X)
                return probs[:, target_class]
            
            # Use KernelExplainer for model-agnostic explanations
            explainer = shap.KernelExplainer(f, baseline.reshape(1, -1))
            shap_values = explainer.shap_values(x.reshape(1, -1), nsamples=100)
            
            return shap_values[0] if isinstance(shap_values[0], np.ndarray) else shap_values[0]
            
        except ImportError:
            self.logger.warning("SHAP not available, falling back to gradient approximation")
            return self._compute_gradient_approximation(model_wrapper, x, target_class, baseline)
    
    def _compute_ig_attributions(self, model_wrapper, x: np.ndarray, target_class: int,
                               baseline: np.ndarray) -> np.ndarray:
        """Compute Integrated Gradients attributions."""
        # Simplified IG implementation
        n_steps = 50
        alphas = np.linspace(0, 1, n_steps)
        
        attributions = np.zeros_like(x)
        
        for alpha in alphas:
            x_interp = baseline + alpha * (x - baseline)
            grad = self._compute_numerical_gradient(model_wrapper, x_interp, target_class)
            attributions += grad
            
        attributions = attributions * (x - baseline) / n_steps
        return attributions
    
    def _compute_lime_attributions(self, model_wrapper, x: np.ndarray, target_class: int) -> np.ndarray:
        """Compute LIME attributions using local perturbations."""
        try:
            from lime.lime_tabular import LimeTabularExplainer
            
            # Create LIME explainer
            explainer = LimeTabularExplainer(
                training_data=x.reshape(1, -1),  # Minimal training data
                feature_names=[f'feature_{i}' for i in range(len(x))],
                discretize_continuous=False
            )
            
            # Get explanation
            def predict_fn(X):
                return model_wrapper.predict_proba(X)
            
            explanation = explainer.explain_instance(
                x, predict_fn, labels=[target_class], num_features=len(x)
            )
            
            # Extract attributions
            attributions = np.zeros_like(x)
            for feature_id, importance in explanation.as_list():
                feature_idx = int(feature_id.split('_')[-1]) if 'feature_' in feature_id else int(feature_id)
                attributions[feature_idx] = importance
                
            return attributions
            
        except ImportError:
            self.logger.warning("LIME not available, falling back to gradient approximation")
            return self._compute_gradient_approximation(model_wrapper, x, target_class, 
                                                      np.zeros_like(x))
    
    def _compute_numerical_gradient(self, model_wrapper, x: np.ndarray, target_class: int) -> np.ndarray:
        """Compute numerical gradients."""
        h = 1e-5
        grad = np.zeros_like(x)
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            
            prob_plus = model_wrapper.predict_proba(x_plus.reshape(1, -1))[0, target_class]
            prob_minus = model_wrapper.predict_proba(x_minus.reshape(1, -1))[0, target_class]
            
            grad[i] = (prob_plus - prob_minus) / (2 * h)
            
        return grad
    
    def _compute_gradient_approximation(self, model_wrapper, x: np.ndarray, target_class: int,
                                      baseline: np.ndarray) -> np.ndarray:
        """Fallback gradient-based attribution."""
        return self._compute_numerical_gradient(model_wrapper, x, target_class)
    
    def _compute_counterfactual_pulls(self, model_wrapper, x: np.ndarray, y_true: int,
                                    negative_indices: np.ndarray, baseline: np.ndarray) -> np.ndarray:
        """
        Compute counterfactual pull for each negative feature.
        
        This measures how much masking each feature increases the true class probability.
        """
        pulls = np.zeros(len(negative_indices))
        
        # Original probability for true class
        prob_original = model_wrapper.predict_proba(x.reshape(1, -1))[0, y_true]
        
        for i, feature_idx in enumerate(negative_indices):
            # Create masked version
            x_masked = x.copy()
            x_masked[feature_idx] = baseline[feature_idx]
            
            # Get probability after masking
            prob_masked = model_wrapper.predict_proba(x_masked.reshape(1, -1))[0, y_true]
            
            # Counterfactual pull is the increase in true class probability
            pulls[i] = prob_masked - prob_original
            
        return pulls
    
    def _compute_a_flip_score(self, attributions_pred: np.ndarray, 
                            attributions_runner_up: np.ndarray) -> float:
        """
        Compute Attribution-Flip (A-Flip) score measuring explanation instability.
        
        From equation (4) in the paper:
        α(x) = 1/2 * Σ|φᵢ(x,f,ŷ) - φᵢ(x,f,y₂)|
        """
        return 0.5 * np.sum(np.abs(attributions_pred - attributions_runner_up))
    
    def analyze_sample(self, model_wrapper, x: Union[np.ndarray, torch.Tensor], 
                      y_true: int, return_dict: bool = False) -> Union[RAResult, Dict]:
        """
        Convenience method to analyze a single sample.
        
        Args:
            model_wrapper: Model wrapper
            x: Input sample
            y_true: True label
            return_dict: Return results as dictionary if True
            
        Returns:
            RAResult or dictionary with analysis results
        """
        result = self.compute_reverse_attribution(model_wrapper, x, y_true)
        
        if return_dict:
            return {
                'counter_evidence': result.counter_evidence,
                'a_flip_score': result.a_flip_score,
                'predicted_class': result.predicted_class,
                'true_class': result.true_class,
                'runner_up_class': result.runner_up_class,
                'confidence_original': result.confidence_original,
                'top_counter_features': [idx for idx, _, _ in result.get_top_counter_evidence()],
                'interpretation': self._generate_interpretation(result)
            }
        
        return result
    
    def _generate_interpretation(self, result: RAResult) -> Dict[str, str]:
        """Generate human-readable interpretation of results."""
        interpretation = {}
        
        # Overall confidence assessment
        if result.a_flip_score > 0.5:
            confidence_level = "low"
        elif result.a_flip_score > 0.2:
            confidence_level = "moderate"
        else:
            confidence_level = "high"
            
        interpretation['confidence_assessment'] = (
            f"Model confidence is {confidence_level} (A-Flip score: {result.a_flip_score:.3f})"
        )
        
        # Counter-evidence summary
        top_features = result.get_top_counter_evidence(3)
        if top_features:
            feature_list = [f"feature {idx} (pull: {pull:.3f})" for idx, _, pull in top_features]
            interpretation['counter_evidence_summary'] = (
                f"Key features suppressing the correct answer: {', '.join(feature_list)}"
            )
        else:
            interpretation['counter_evidence_summary'] = "No significant counter-evidence found"
            
        # Prediction analysis
        if result.predicted_class != result.true_class:
            interpretation['prediction_analysis'] = (
                f"Model incorrectly predicted class {result.predicted_class} instead of {result.true_class}. "
                f"Counter-evidence features may be responsible for this misclassification."
            )
        else:
            interpretation['prediction_analysis'] = (
                f"Model correctly predicted class {result.predicted_class}, but with potential hesitation "
                f"due to counter-evidence features."
            )
            
        return interpretation
    
    def batch_analyze(self, model_wrapper, X: np.ndarray, y_true: np.ndarray,
                     progress_callback=None) -> List[RAResult]:
        """
        Analyze multiple samples in batch.
        
        Args:
            model_wrapper: Model wrapper
            X: Input samples (n_samples, n_features)
            y_true: True labels
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of RAResult objects
        """
        results = []
        
        for i, (x, y) in enumerate(zip(X, y_true)):
            if progress_callback:
                progress_callback(i, len(X))
                
            try:
                result = self.compute_reverse_attribution(model_wrapper, x, y)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error analyzing sample {i}: {str(e)}")
                # Create empty result for failed samples
                results.append(RAResult(
                    counter_evidence=[],
                    a_flip_score=0.0,
                    negative_attributions=np.array([]),
                    counterfactual_pulls=np.array([]),
                    predicted_class=-1,
                    true_class=y,
                    runner_up_class=-1,
                    confidence_original=0.0
                ))
                
        return results
'''

# Write the core RA implementation
with open('reverse-attribution/core/ra.py', 'w') as f:
    f.write(core_ra_code)

print("✅ Core RA algorithm implementation completed!")
print("Features implemented:")
print("- Complete 4-step RA algorithm as per JMLR paper")
print("- SHAP, Integrated Gradients, and LIME attribution support")
print("- Counterfactual pull calculations")  
print("- Attribution-Flip (A-Flip) score computation")
print("- Batch processing capabilities")
print("- Comprehensive result container (RAResult)")
print("- Human-readable interpretations")