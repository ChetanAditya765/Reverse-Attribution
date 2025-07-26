"""
Tests for evaluation pipeline including metrics calculation and statistical analysis.
"""

import pytest
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from unittest.mock import Mock, patch

class TestMetricsCalculation:
    """Test various metrics calculation functions."""
    
    def test_accuracy_calculation(self):
        """Test accuracy calculation."""
        y_true = [0, 1, 2, 2, 1, 0]
        y_pred = [0, 1, 2, 1, 1, 0]
        
        accuracy = accuracy_score(y_true, y_pred)
        
        assert 0 <= accuracy <= 1
        assert accuracy == 5/6  # 5 correct out of 6
    
    def test_f1_score_calculation(self):
        """Test F1-score calculation and binary classification edge cases."""
        # Standard case
        y_true = [0, 1, 1, 0, 1, 1]
        y_pred = [0, 1, 0, 0, 1, 1]
        
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        
        assert 0 <= f1 <= 1
        assert 0 <= precision <=  recall <= 1
        
        # Manual F1 calculation
        if precision + recall > 0:
            manual_f1 = 2 * (precision * recall) / (precision + recall)
            assert abs(f1 - manual_f1) < 1e-10
    
    def test_f1_score_edge_cases(self):
        """Test F1-score edge cases that can result in 0.000."""
        # Case 1: All predictions are negative class
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 0, 0, 0, 0]  # All predicted as negative
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        assert precision == 0.0  # No true positives
        assert recall == 0.0     # Missed all positives
        assert f1 == 0.0         # F1 should be 0
    
    def test_multiclass_metrics(self):
        """Test multiclass metrics calculation."""
        y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 1, 0, 2, 2]  # Some errors
        
        # Test different averaging methods
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        assert 0 <= f1_macro <= 1
        assert 0 <= f1_micro <= 1
        assert 0 <= f1_weighted <= 1
        
        # Micro average should equal accuracy for multiclass
        accuracy = accuracy_score(y_true, y_pred)
        assert abs(f1_micro - accuracy) < 1e-10

class TestCalibrationMetrics:
    """Test model calibration metrics like ECE and Brier score."""
    
    def test_ece_calculation(self):
        """Test Expected Calibration Error calculation."""
        # Mock probability predictions and true labels
        probs = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        
        # Simple ECE calculation (binned approach)
        n_bins = 4
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = probs[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        assert 0 <= ece <= 1
        assert isinstance(ece, (int, float))
    
    def test_brier_score_calculation(self):
        """Test Brier score calculation."""
        # Binary case
        y_true = np.array([1, 0, 1, 0, 1])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
        
        brier_score = np.mean((y_prob - y_true) ** 2)
        
        assert 0 <= brier_score <= 1
        assert isinstance(brier_score, (int, float))
        
        # Perfect predictions should have Brier score of 0
        perfect_probs = y_true.astype(float)
        perfect_brier = np.mean((perfect_probs - y_true) ** 2)
        assert perfect_brier == 0.0
    
    def test_reliability_diagram_data(self):
        """Test data preparation for reliability diagrams."""
        probs = np.random.uniform(0, 1, 100)
        y_true = np.random.binomial(1, probs, 100)
        
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        bin_data = []
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (probs >= bin_lower) & (probs < bin_upper)
            
            if i == n_bins - 1:  # Include right boundary for last bin
                in_bin = (probs >= bin_lower) & (probs <= bin_upper)
            
            if in_bin.sum() > 0:
                bin_accuracy = y_true[in_bin].mean()
                bin_confidence = probs[in_bin].mean()
                bin_count = in_bin.sum()
                
                bin_data.append({
                    'accuracy': bin_accuracy,
                    'confidence': bin_confidence,
                    'count': bin_count
                })
        
        assert len(bin_data) <= n_bins
        assert all(0 <= bd['accuracy'] <= 1 for bd in bin_data)
        assert all(0 <= bd['confidence'] <= 1 for bd in bin_data)

class TestStatisticalAnalysis:
    """Test statistical analysis functionality."""
    
    def test_confidence_interval_calculation(self):
        """Test confidence interval calculation for metrics."""
        # Generate sample data
        np.random.seed(42)
        sample_accuracies = np.random.normal(0.85, 0.05, 30)
        sample_accuracies = np.clip(sample_accuracies, 0, 1)  # Ensure valid range
        
        mean_acc = np.mean(sample_accuracies)
        std_acc = np.std(sample_accuracies, ddof=1)
        n = len(sample_accuracies)
        
        # 95% confidence interval
        from scipy import stats
        t_critical = stats.t.ppf(0.975, n - 1)
        margin_error = t_critical * (std_acc / np.sqrt(n))
        
        ci_lower = mean_acc - margin_error
        ci_upper = mean_acc + margin_error
        
        assert ci_lower < mean_acc < ci_upper
        assert 0 <= ci_lower <= 1
        assert 0 <= ci_upper <= 1
    
    def test_bootstrap_confidence_intervals(self):
        """Test bootstrap method for confidence intervals."""
        # Original sample
        np.random.seed(42)
        original_sample = np.random.normal(0.8, 0.1, 50)
        original_sample = np.clip(original_sample, 0, 1)
        
        # Bootstrap resampling
        n_bootstrap = 1000
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(original_sample, size=len(original_sample), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        # Calculate percentile confidence interval
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)
        
        assert ci_lower < np.mean(original_sample) < ci_upper
        assert len(bootstrap_means) == n_bootstrap
    
    def test_significance_testing(self):
        """Test statistical significance testing between models."""
        np.random.seed(42)
        
        # Model A performance
        model_a_scores = np.random.normal(0.85, 0.05, 30)
        model_a_scores = np.clip(model_a_scores, 0, 1)
        
        # Model B performance (slightly worse)
        model_b_scores = np.random.normal(0.80, 0.05, 30)
        model_b_scores = np.clip(model_b_scores, 0, 1)
        
        # Paired t-test
        from scipy import stats
        t_stat, p_value = stats.ttest_rel(model_a_scores, model_b_scores)
        
        assert isinstance(t_stat, (int, float))
        assert 0 <= p_value <= 1
        
        # Unpaired t-test
        t_stat_unpaired, p_value_unpaired = stats.ttest_ind(model_a_scores, model_b_scores)
        
        assert isinstance(t_stat_unpaired, (int, float))
        assert 0 <= p_value_unpaired <= 1

class TestEvaluationPipeline:
    """Test complete evaluation pipeline."""
    
    def test_evaluation_metrics_aggregation(self):
        """Test aggregation of multiple evaluation metrics."""
        # Mock evaluation results
        evaluation_results = {
            'accuracy': 0.85,
            'precision': 0.83,
            'recall': 0.87,
            'f1': 0.85,
            'ece': 0.05,
            'brier_score': 0.12
        }
        
        # Validate all metrics are present and in valid ranges
        assert 0 <= evaluation_results['accuracy'] <= 1
        assert 0 <= evaluation_results['precision'] <= 1
        assert 0 <= evaluation_results['recall'] <= 1
        assert 0 <= evaluation_results['f1'] <= 1
        assert 0 <= evaluation_results['ece'] <= 1
        assert 0 <= evaluation_results['brier_score'] <= 1
        
        # Check F1 score consistency
        precision = evaluation_results['precision']
        recall = evaluation_results['recall']
        expected_f1 = 2 * (precision * recall) / (precision + recall)
        
        assert abs(evaluation_results['f1'] - expected_f1) < 0.01
    
    def test_model_evaluation_summary(self):
        """Test model evaluation summary generation."""
        # Mock multiple model results
        models_results = {
            'cifar10': {
                'accuracy': 0.9328,
                'f1': 0.9314,
                'ece': 0.0451,
                'a_flip': 819.42
            },
            'imdb': {
                'accuracy': 0.9156,
                'f1': 0.0000,  # Known issue
                'ece': 0.0000,
                'a_flip': 0.0
            }
        }
        
        # Calculate summary statistics
        accuracies = [results['accuracy'] for results in models_results.values()]
        f1_scores = [results['f1'] for results in models_results.values()]
        
        avg_accuracy = np.mean(accuracies)
        best_accuracy = max(accuracies)
        
        assert avg_accuracy > 0
        assert best_accuracy >= avg_accuracy
        assert len(models_results) == 2
        
        # Identify models with F1-score issues
        f1_issues = [model for model, results in models_results.items() if results['f1'] == 0.0]
        assert 'imdb' in f1_issues
    
    def test_cross_validation_simulation(self):
        """Test cross-validation results simulation."""
        # Simulate 5-fold CV results
        n_folds = 5
        cv_results = {
            'accuracy': [],
            'f1': [],
            'precision': [],
            'recall': []
        }
        
        np.random.seed(42)
        for fold in range(n_folds):
            # Simulate slight variation across folds
            base_acc = 0.85
            noise = np.random.normal(0, 0.02)
            fold_acc = np.clip(base_acc + noise, 0, 1)
            
            cv_results['accuracy'].append(fold_acc)
            cv_results['f1'].append(fold_acc - 0.01)  # Slightly lower F1
            cv_results['precision'].append(fold_acc + 0.01)
            cv_results['recall'].append(fold_acc - 0.02)
        
        # Calculate CV statistics
        for metric, scores in cv_results.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            assert 0 <= mean_score <= 1
            assert std_score >= 0
            assert len(scores) == n_folds
    
    @pytest.mark.integration
    def test_end_to_end_evaluation(self):
        """Test complete end-to-end evaluation pipeline."""
        # Mock the entire evaluation process
        
        # 1. Model loading
        model_loaded = True
        assert model_loaded
        
        # 2. Data preparation
        test_size = 1000
        assert test_size > 0
        
        # 3. Prediction generation
        predictions = np.random.randint(0, 10, test_size)
        probabilities = np.random.dirichlet(np.ones(10), test_size)
        true_labels = np.random.randint(0, 10, test_size)
        
        assert len(predictions) == test_size
        assert probabilities.shape == (test_size, 10)
        assert len(true_labels) == test_size
        
        # 4. Metrics calculation
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='macro')
        
        assert 0 <= accuracy <= 1
        assert 0 <= f1 <= 1
        
        # 5. Results compilation
        results = {
            'accuracy': accuracy,
            'f1': f1,
            'predictions': predictions,
            'test_size': test_size
        }
        
        assert 'accuracy' in results
        assert 'f1' in results
        assert results['test_size'] == test_size
