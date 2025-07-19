"""
Comprehensive metrics for evaluating model performance and explanations.
Includes the 4 specific metrics from the Reverse Attribution JMLR paper:
1. Misprediction Localization (Jaccard/F1)
2. Debugging Time 
3. Confidence Calibration (ECE)
4. Trust Change Assessment
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any, Union
from sklearn.metrics import brier_score_loss, log_loss
import scipy.stats as stats
from scipy.spatial.distance import cosine
import warnings


# =============================================================================
# JMLR Paper Metric 1: Misprediction Localization
# =============================================================================

def jaccard_index(set_pred: set, set_true: set) -> float:
    """
    Compute Jaccard Index between predicted and true feature sets.
    
    Args:
        set_pred: Set of predicted important feature indices
        set_true: Set of ground truth important feature indices
        
    Returns:
        Jaccard index (intersection over union)
    """
    if len(set_pred) == 0 and len(set_true) == 0:
        return 1.0
    
    intersection = len(set_pred & set_true)
    union = len(set_pred | set_true)
    
    return intersection / union if union > 0 else 0.0


def f1_score_localization(pred_indices: List[int], true_indices: List[int]) -> float:
    """
    Compute F1 score for localization task.
    
    Args:
        pred_indices: Predicted important feature indices
        true_indices: Ground truth important feature indices
        
    Returns:
        F1 score for localization
    """
    set_pred = set(pred_indices)
    set_true = set(true_indices)
    
    if len(set_pred) == 0 and len(set_true) == 0:
        return 1.0
    
    if len(set_pred) == 0 or len(set_true) == 0:
        return 0.0
    
    tp = len(set_pred & set_true)  # True positives
    fp = len(set_pred - set_true)  # False positives  
    fn = len(set_true - set_pred)  # False negatives
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def compute_localization_metrics(
    ra_results: List[Dict[str, Any]], 
    ground_truth_masks: Dict[int, List[int]],
    top_k: int = 10
) -> Dict[str, float]:
    """
    Compute localization metrics for multiple samples.
    
    Args:
        ra_results: List of RA explanation results
        ground_truth_masks: Ground truth feature importance masks
        top_k: Number of top features to consider
        
    Returns:
        Dictionary with localization metrics
    """
    jaccard_scores = []
    f1_scores = []
    
    for i, ra_result in enumerate(ra_results):
        if i in ground_truth_masks:
            # Extract top-k counter-evidence features
            counter_evidence = ra_result.get('counter_evidence', [])
            if counter_evidence:
                pred_indices = [ce[0] for ce in counter_evidence[:top_k]]
            else:
                pred_indices = []
            
            true_indices = ground_truth_masks[i]
            
            # Compute metrics
            jaccard = jaccard_index(set(pred_indices), set(true_indices))
            f1 = f1_score_localization(pred_indices, true_indices)
            
            jaccard_scores.append(jaccard)
            f1_scores.append(f1)
    
    return {
        'avg_jaccard': np.mean(jaccard_scores) if jaccard_scores else 0.0,
        'std_jaccard': np.std(jaccard_scores) if jaccard_scores else 0.0,
        'avg_f1': np.mean(f1_scores) if f1_scores else 0.0,
        'std_f1': np.std(f1_scores) if f1_scores else 0.0,
        'num_samples': len(jaccard_scores)
    }


# =============================================================================
# JMLR Paper Metric 2: Debugging Time
# =============================================================================

def average_debug_time(timings: List[float]) -> float:
    """
    Compute average debugging time.
    
    Args:
        timings: List of debugging times in seconds
        
    Returns:
        Average debugging time
    """
    return sum(timings) / len(timings) if timings else 0.0


def debug_time_improvement(
    times_with_ra: List[float], 
    times_without_ra: List[float]
) -> Dict[str, float]:
    """
    Compute debugging time improvement when using RA.
    
    Args:
        times_with_ra: Debugging times with RA explanations
        times_without_ra: Debugging times without RA explanations
        
    Returns:
        Dictionary with improvement metrics
    """
    avg_with = average_debug_time(times_with_ra)
    avg_without = average_debug_time(times_without_ra)
    
    improvement_seconds = avg_without - avg_with
    improvement_percent = (improvement_seconds / avg_without * 100) if avg_without > 0 else 0.0
    
    # Statistical significance test
    if len(times_with_ra) > 1 and len(times_without_ra) > 1:
        statistic, p_value = stats.ttest_ind(times_without_ra, times_with_ra)
        is_significant = p_value < 0.05
    else:
        p_value = 1.0
        is_significant = False
    
    return {
        'avg_time_with_ra': avg_with,
        'avg_time_without_ra': avg_without,
        'improvement_seconds': improvement_seconds,
        'improvement_percent': improvement_percent,
        'p_value': p_value,
        'is_significant': is_significant
    }


def debug_success_rate(
    success_with_ra: List[bool], 
    success_without_ra: List[bool]
) -> Dict[str, float]:
    """
    Compute debugging success rates.
    
    Args:
        success_with_ra: Boolean list of debugging success with RA
        success_without_ra: Boolean list of debugging success without RA
        
    Returns:
        Dictionary with success rate metrics
    """
    rate_with = np.mean(success_with_ra) if success_with_ra else 0.0
    rate_without = np.mean(success_without_ra) if success_without_ra else 0.0
    
    return {
        'success_rate_with_ra': rate_with,
        'success_rate_without_ra': rate_without,
        'success_improvement': rate_with - rate_without
    }


# =============================================================================
# JMLR Paper Metric 3: Confidence Calibration (ECE)
# =============================================================================

def expected_calibration_error(
    confidences: np.ndarray, 
    accuracies: np.ndarray, 
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE) as described in the JMLR paper.
    
    Args:
        confidences: Model confidence scores [N]
        accuracies: Binary accuracy per sample [N] (0 or 1)
        n_bins: Number of bins for calibration
        
    Returns:
        ECE score
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.sum() / len(confidences)
        
        if prop_in_bin > 0:
            # Accuracy in this bin
            accuracy_in_bin = accuracies[in_bin].mean()
            
            # Average confidence in this bin
            avg_confidence_in_bin = confidences[in_bin].mean()
            
            # Add to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def maximum_calibration_error(
    confidences: np.ndarray, 
    accuracies: np.ndarray, 
    n_bins: int = 10
) -> float:
    """
    Compute Maximum Calibration Error (MCE).
    
    Args:
        confidences: Model confidence scores [N]
        accuracies: Binary accuracy per sample [N] (0 or 1)
        n_bins: Number of bins
        
    Returns:
        MCE score
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    calibration_errors = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        
        if in_bin.sum() > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            calibration_errors.append(abs(avg_confidence_in_bin - accuracy_in_bin))
    
    return max(calibration_errors) if calibration_errors else 0.0


def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Brier Score for calibration assessment.
    
    Args:
        probs: Predicted probabilities [N, num_classes]
        labels: True labels [N]
        
    Returns:
        Brier score
    """
    if probs.ndim == 1:
        # Binary case
        return brier_score_loss(labels, probs)
    else:
        # Multi-class case - average over classes
        n_classes = probs.shape[1]
        brier_scores = []
        
        for class_idx in range(n_classes):
            true_class = (labels == class_idx).astype(int)
            pred_class = probs[:, class_idx]
            brier_scores.append(brier_score_loss(true_class, pred_class))
        
        return np.mean(brier_scores)


def post_intervention_calibration(
    original_probs: np.ndarray,
    intervention_probs: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    Compare calibration before and after RA-guided interventions.
    
    Args:
        original_probs: Original model probabilities
        intervention_probs: Probabilities after intervention
        labels: True labels
        
    Returns:
        Calibration comparison metrics
    """
    # Extract confidences
    orig_conf = np.max(original_probs, axis=1)
    interv_conf = np.max(intervention_probs, axis=1)
    
    # Extract predictions
    orig_pred = np.argmax(original_probs, axis=1)
    interv_pred = np.argmax(intervention_probs, axis=1)
    
    # Compute accuracies
    orig_acc = (orig_pred == labels).astype(int)
    interv_acc = (interv_pred == labels).astype(int)
    
    # Compute ECE for both
    orig_ece = expected_calibration_error(orig_conf, orig_acc)
    interv_ece = expected_calibration_error(interv_conf, interv_acc)
    
    return {
        'original_ece': orig_ece,
        'intervention_ece': interv_ece,
        'ece_improvement': orig_ece - interv_ece,
        'original_accuracy': np.mean(orig_acc),
        'intervention_accuracy': np.mean(interv_acc)
    }


# =============================================================================
# JMLR Paper Metric 4: Trust Change Assessment
# =============================================================================

def trust_change(
    before_scores: List[float], 
    after_scores: List[float]
) -> float:
    """
    Compute average trust change after seeing RA explanations.
    
    Args:
        before_scores: Trust scores before explanation (e.g., 1-5 Likert scale)
        after_scores: Trust scores after explanation
        
    Returns:
        Average trust change (positive = increased trust)
    """
    if len(before_scores) != len(after_scores):
        raise ValueError("Before and after scores must have same length")
    
    changes = [after - before for before, after in zip(before_scores, after_scores)]
    return np.mean(changes)


def trust_calibration_analysis(
    user_trust_scores: np.ndarray,
    model_accuracy_scores: np.ndarray,
    n_bins: int = 5
) -> Dict[str, float]:
    """
    Analyze calibration between human trust and model accuracy.
    
    Args:
        user_trust_scores: Human trust ratings [N] (e.g., 1-5 scale, normalized to 0-1)
        model_accuracy_scores: Binary accuracy per sample [N] (0 or 1)
        n_bins: Number of bins for trust analysis
        
    Returns:
        Trust calibration metrics
    """
    if len(user_trust_scores) != len(model_accuracy_scores):
        raise ValueError("Trust scores and accuracy scores must have same length")
    
    # Normalize trust scores to 0-1 if needed
    if np.max(user_trust_scores) > 1.0:
        trust_normalized = (user_trust_scores - np.min(user_trust_scores)) / (np.max(user_trust_scores) - np.min(user_trust_scores))
    else:
        trust_normalized = user_trust_scores
    
    # Create bins based on trust scores
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    trust_calibration_error = 0.0
    valid_bins = 0
    
    bin_data = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (trust_normalized >= bin_lower) & (trust_normalized < bin_upper)
        if i == n_bins - 1:  # Include upper boundary in last bin
            in_bin = (trust_normalized >= bin_lower) & (trust_normalized <= bin_upper)
        
        count = in_bin.sum()
        
        if count > 0:
            avg_trust = trust_normalized[in_bin].mean()
            avg_accuracy = model_accuracy_scores[in_bin].mean()
            
            trust_calibration_error += abs(avg_trust - avg_accuracy) * (count / len(user_trust_scores))
            valid_bins += 1
            
            bin_data.append({
                'bin_range': (bin_lower, bin_upper),
                'avg_trust': avg_trust,
                'avg_accuracy': avg_accuracy,
                'count': count
            })
    
    # Compute correlation
    trust_accuracy_correlation = np.corrcoef(trust_normalized, model_accuracy_scores)[0, 1]
    if np.isnan(trust_accuracy_correlation):
        trust_accuracy_correlation = 0.0
    
    return {
        'trust_calibration_error': trust_calibration_error,
        'trust_accuracy_correlation': trust_accuracy_correlation,
        'valid_bins': valid_bins,
        'bin_data': bin_data
    }


def trust_improvement_significance(
    before_scores: List[float],
    after_scores: List[float],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Test statistical significance of trust improvement.
    
    Args:
        before_scores: Trust scores before explanation
        after_scores: Trust scores after explanation  
        alpha: Significance level
        
    Returns:
        Statistical test results
    """
    if len(before_scores) != len(after_scores):
        raise ValueError("Before and after scores must have same length")
    
    # Paired t-test
    statistic, p_value = stats.ttest_rel(after_scores, before_scores)
    is_significant = p_value < alpha
    
    # Effect size (Cohen's d for paired samples)
    differences = np.array(after_scores) - np.array(before_scores)
    effect_size = np.mean(differences) / np.std(differences) if np.std(differences) > 0 else 0.0
    
    return {
        'mean_change': np.mean(differences),
        'std_change': np.std(differences),
        'statistic': statistic,
        'p_value': p_value,
        'is_significant': is_significant,
        'effect_size': effect_size,
        'improvement_rate': np.mean(differences > 0)
    }


# =============================================================================
# RA-Specific Metrics
# =============================================================================

def compute_ra_instability_metrics(
    ra_results: List[Dict[str, Any]],
    model_predictions: np.ndarray,
    true_labels: np.ndarray
) -> Dict[str, float]:
    """
    Compute RA-specific instability metrics from the paper.
    
    Args:
        ra_results: List of RA results from ReverseAttribution.explain()
        model_predictions: Model predictions [N]
        true_labels: True labels [N]
        
    Returns:
        RA instability metrics
    """
    if not ra_results:
        return {'error': 'No RA results provided'}
    
    # Extract A-Flip scores
    a_flip_scores = [result.get('a_flip', 0) for result in ra_results]
    counter_evidence_counts = [len(result.get('counter_evidence', [])) for result in ra_results]
    counter_evidence_strengths = []
    
    for result in ra_results:
        ce = result.get('counter_evidence', [])
        if ce:
            # Average absolute suppression strength
            avg_strength = np.mean([abs(item[2]) for item in ce])
            counter_evidence_strengths.append(avg_strength)
        else:
            counter_evidence_strengths.append(0.0)
    
    # Separate by correct/incorrect predictions
    correct_mask = (model_predictions == true_labels)
    incorrect_mask = ~correct_mask
    
    metrics = {
        # Overall A-Flip statistics
        'avg_a_flip': np.mean(a_flip_scores),
        'std_a_flip': np.std(a_flip_scores),
        'median_a_flip': np.median(a_flip_scores),
        
        # Counter-evidence statistics  
        'avg_counter_evidence_count': np.mean(counter_evidence_counts),
        'avg_counter_evidence_strength': np.mean(counter_evidence_strengths),
        'std_counter_evidence_strength': np.std(counter_evidence_strengths),
        
        # Samples with counter-evidence
        'pct_samples_with_counter_evidence': np.mean([count > 0 for count in counter_evidence_counts]) * 100,
        
        # High instability samples (A-Flip > threshold)
        'pct_high_instability': np.mean([score > 0.5 for score in a_flip_scores]) * 100,
    }
    
    # Stratified analysis by prediction correctness
    if np.sum(correct_mask) > 0:
        metrics.update({
            'avg_a_flip_correct': np.mean([score for i, score in enumerate(a_flip_scores) if correct_mask[i]]),
            'avg_ce_count_correct': np.mean([count for i, count in enumerate(counter_evidence_counts) if correct_mask[i]]),
        })
    
    if np.sum(incorrect_mask) > 0:
        metrics.update({
            'avg_a_flip_incorrect': np.mean([score for i, score in enumerate(a_flip_scores) if incorrect_mask[i]]),
            'avg_ce_count_incorrect': np.mean([count for i, count in enumerate(counter_evidence_counts) if incorrect_mask[i]]),
        })
        
        # Effect size of A-Flip difference between correct/incorrect
        if np.sum(correct_mask) > 0 and np.sum(incorrect_mask) > 0:
            correct_scores = [score for i, score in enumerate(a_flip_scores) if correct_mask[i]]
            incorrect_scores = [score for i, score in enumerate(a_flip_scores) if incorrect_mask[i]]
            
            # Cohen's d effect size
            pooled_std = np.sqrt(((len(correct_scores) - 1) * np.var(correct_scores) + 
                                 (len(incorrect_scores) - 1) * np.var(incorrect_scores)) / 
                                (len(correct_scores) + len(incorrect_scores) - 2))
            
            if pooled_std > 0:
                cohens_d = (np.mean(incorrect_scores) - np.mean(correct_scores)) / pooled_std
                metrics['a_flip_effect_size'] = cohens_d
    
    return metrics


# =============================================================================
# Comprehensive Evaluation Function
# =============================================================================

def evaluate_all_jmlr_metrics(
    model: torch.nn.Module,
    ra_results: List[Dict[str, Any]],
    ground_truth_masks: Dict[int, List[int]],
    user_study_data: Dict[str, List[float]],
    model_predictions: np.ndarray,
    model_probabilities: np.ndarray,
    true_labels: np.ndarray
) -> Dict[str, Any]:
    """
    Compute all 4 JMLR paper metrics in one comprehensive function.
    
    Args:
        model: PyTorch model
        ra_results: RA explanation results
        ground_truth_masks: Ground truth importance masks
        user_study_data: User study results
        model_predictions: Model predictions
        model_probabilities: Model probability outputs
        true_labels: Ground truth labels
        
    Returns:
        Comprehensive metrics dictionary
    """
    all_metrics = {}
    
    # Metric 1: Misprediction Localization
    print("Computing misprediction localization metrics...")
    localization_metrics = compute_localization_metrics(
        ra_results, ground_truth_masks, top_k=10
    )
    all_metrics['localization'] = localization_metrics
    
    # Metric 2: Debugging Time
    if 'debug_times_with_ra' in user_study_data and 'debug_times_without_ra' in user_study_data:
        print("Computing debugging time metrics...")
        debug_metrics = debug_time_improvement(
            user_study_data['debug_times_with_ra'],
            user_study_data['debug_times_without_ra']
        )
        all_metrics['debugging'] = debug_metrics
    
    # Metric 3: Confidence Calibration
    print("Computing calibration metrics...")
    confidences = np.max(model_probabilities, axis=1)
    correct_predictions = (model_predictions == true_labels).astype(int)
    
    calibration_metrics = {
        'ece': expected_calibration_error(confidences, correct_predictions),
        'mce': maximum_calibration_error(confidences, correct_predictions),
        'brier_score': compute_brier_score(model_probabilities, true_labels)
    }
    all_metrics['calibration'] = calibration_metrics
    
    # Metric 4: Trust Change Assessment
    if 'trust_before' in user_study_data and 'trust_after' in user_study_data:
        print("Computing trust metrics...")
        trust_metrics = {
            'avg_trust_change': trust_change(
                user_study_data['trust_before'],
                user_study_data['trust_after']
            ),
            'trust_calibration': trust_calibration_analysis(
                np.array(user_study_data['trust_after']),
                correct_predictions[:len(user_study_data['trust_after'])]
            ),
            'significance_test': trust_improvement_significance(
                user_study_data['trust_before'],
                user_study_data['trust_after']
            )
        }
        all_metrics['trust'] = trust_metrics
    
    # Additional RA-specific metrics
    print("Computing RA-specific metrics...")
    ra_metrics = compute_ra_instability_metrics(ra_results, model_predictions, true_labels)
    all_metrics['ra_instability'] = ra_metrics
    
    return all_metrics


if __name__ == "__main__":
    # Example usage and testing
    print("Testing JMLR metrics implementation...")
    
    # Generate synthetic data for testing
    np.random.seed(42)
    n_samples = 100
    
    # Test Metric 1: Localization
    print("\n1. Testing localization metrics...")
    pred_indices = [1, 3, 5, 7]
    true_indices = [1, 2, 5, 8, 9]
    
    jaccard = jaccard_index(set(pred_indices), set(true_indices))
    f1 = f1_score_localization(pred_indices, true_indices)
    print(f"   Jaccard Index: {jaccard:.3f}")
    print(f"   F1 Score: {f1:.3f}")
    
    # Test Metric 2: Debugging Time
    print("\n2. Testing debugging time metrics...")
    times_with_ra = [45.2, 38.1, 42.5, 39.8, 41.2]
    times_without_ra = [65.3, 58.7, 62.1, 59.4, 60.8]
    
    debug_metrics = debug_time_improvement(times_with_ra, times_without_ra)
    print(f"   Time improvement: {debug_metrics['improvement_seconds']:.1f}s ({debug_metrics['improvement_percent']:.1f}%)")
    print(f"   Significant: {debug_metrics['is_significant']}")
    
    # Test Metric 3: Calibration
    print("\n3. Testing calibration metrics...")
    confidences = np.random.beta(2, 2, n_samples)
    accuracies = np.random.binomial(1, confidences * 0.8, n_samples)  # Slightly miscalibrated
    
    ece = expected_calibration_error(confidences, accuracies)
    mce = maximum_calibration_error(confidences, accuracies)
    print(f"   ECE: {ece:.4f}")
    print(f"   MCE: {mce:.4f}")
    
    # Test Metric 4: Trust
    print("\n4. Testing trust metrics...")
    trust_before = [2.1, 2.8, 3.2, 2.5, 3.1, 2.9, 2.7, 3.0, 2.6, 3.3]
    trust_after = [3.2, 3.5, 3.8, 3.1, 3.9, 3.6, 3.4, 3.7, 3.3, 4.1]
    
    avg_change = trust_change(trust_before, trust_after)
    trust_sig = trust_improvement_significance(trust_before, trust_after)
    print(f"   Average trust change: {avg_change:.2f}")
    print(f"   Significant improvement: {trust_sig['is_significant']}")
    print(f"   Effect size: {trust_sig['effect_size']:.3f}")
    
    print("\n✅ All JMLR metrics tests completed successfully!")
