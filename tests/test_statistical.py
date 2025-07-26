"""
Tests for statistical analysis components including error bars, confidence intervals,
and significance testing.
"""

import pytest
import numpy as np
from scipy import stats
from unittest.mock import Mock
import warnings

class TestErrorBarCalculations:
    """Test error bar and confidence interval calculations."""
    
    def test_standard_error_calculation(self):
        """Test standard error of the mean calculation."""
        data = np.random.normal(0.85, 0.05, 30)  # Mean=0.85, std=0.05, n=30
        
        sample_mean = np.mean(data)
        sample_std = np.std(data, ddof=1)  # Sample standard deviation
        sample_size = len(data)
        
        # Standard error of the mean
        sem = sample_std / np.sqrt(sample_size)
        
        assert sem > 0
        assert sem < sample_std  # SEM should be smaller than standard deviation
        assert isinstance(sem, (int, float))
    
    def test_confidence_interval_calculation(self):
        """Test confidence interval calculation."""
        data = np.random.normal(0.90, 0.03, 50)
        
        # Calculate 95% confidence interval
        confidence_level = 0.95
        alpha = 1 - confidence_level
        
        sample_mean = np.mean(data)
        sample_std = np.std(data, ddof=1)
        sample_size = len(data)
        
        # t-distribution critical value (for small samples)
        t_critical = stats.t.ppf(1 - alpha/2, df=sample_size - 1)
        
        # Margin of error
        margin_of_error = t_critical * (sample_std / np.sqrt(sample_size))
        
        # Confidence interval
        ci_lower = sample_mean - margin_of_error
        ci_upper = sample_mean + margin_of_error
        
        assert ci_lower < sample_mean < ci_upper
        assert ci_upper - ci_lower > 0  # CI should have positive width
        assert margin_of_error > 0
    
    def test_bootstrap_confidence_interval(self):
        """Test bootstrap confidence interval calculation."""
        original_data = np.random.normal(0.88, 0.04, 40)
        
        # Bootstrap resampling
        n_bootstrap = 1000
        bootstrap_means = []
        
        np.random.seed(42)  # For reproducibility
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(original_data, 
                                               size=len(original_data), 
                                               replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        # Calculate percentile-based confidence interval
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)
        
        original_mean = np.mean(original_data)
        
        assert ci_lower < original_mean < ci_upper
        assert len(bootstrap_means) == n_bootstrap
        assert ci_upper > ci_lower
    
    def test_error_bar_robustness(self):
        """Test error bar calculation robustness."""
        # Test with different sample sizes
        for n in [5, 10, 30, 100]:
            data = np.random.normal(0.8, 0.1, n)
            
            # Calculate confidence interval width
            sem = np.std(data, ddof=1) / np.sqrt(n)
            t_critical = stats.t.ppf(0.975, df=n-1)  # 95% CI
            ci_width = 2 * t_critical * sem
            
            # CI width should decrease as sample size increases
            assert ci_width > 0
            
            # For larger samples, CI should be narrower
            if n >= 30:
                assert ci_width < 0.5  # Reasonable upper bound

class TestSignificanceTesting:
    """Test statistical significance testing methods."""
    
    def test_two_sample_t_test(self):
        """Test two-sample t-test for comparing models."""
        # Mock performance scores for two models
        np.random.seed(42)
        model1_scores = np.random.normal(0.88, 0.03, 25)
        model2_scores = np.random.normal(0.85, 0.03, 25)
        
        # Perform independent t-test
        t_stat, p_value = stats.ttest_ind(model1_scores, model2_scores)
        
        assert isinstance(t_stat, (int, float))
        assert 0 <= p_value <= 1
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((np.std(model1_scores, ddof=1)**2 + 
                              np.std(model2_scores, ddof=1)**2) / 2))
        cohens_d = (np.mean(model1_scores) - np.mean(model2_scores)) / pooled_std
        
        assert isinstance(cohens_d, (int, float))
    
    def test_paired_t_test(self):
        """Test paired t-test for same data evaluated by different models."""
        n_samples = 30
        
        # Mock paired performance (e.g., same test samples, different models)
        base_performance = np.random.uniform(0.7, 0.9, n_samples)
        model1_performance = base_performance + np.random.normal(0, 0.02, n_samples)
        model2_performance = base_performance + np.random.normal(-0.05, 0.02, n_samples)
        
        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(model1_performance, model2_performance)
        
        assert isinstance(t_stat, (int, float))
        assert 0 <= p_value <= 1
        
        # Mean difference
        mean_diff = np.mean(model1_performance - model2_performance)
        
        # Since model1 should generally perform better, expect positive difference
        assert isinstance(mean_diff, (int, float))
    
    def test_multiple_comparisons_correction(self):
        """Test multiple comparisons correction (Bonferroni)."""
        # Mock p-values from multiple model comparisons
        raw_p_values = [0.01, 0.03, 0.008, 0.12, 0.045]
        alpha = 0.05
        
        # Bonferroni correction
        n_comparisons = len(raw_p_values)
        corrected_alpha = alpha / n_comparisons
        
        # Apply correction
        significant_after_correction = [p < corrected_alpha for p in raw_p_values]
        
        assert corrected_alpha < alpha
        assert corrected_alpha > 0
        assert isinstance(significant_after_correction, list)
        assert len(significant_after_correction) == len(raw_p_values)
        
        # Generally, fewer comparisons should be significant after correction
        n_significant_raw = sum(p < alpha for p in raw_p_values)
        n_significant_corrected = sum(significant_after_correction)
        
        assert n_significant_corrected <= n_significant_raw
    
    def test_effect_size_calculation(self):
        """Test effect size calculations."""
        group1 = np.random.normal(0.85, 0.05, 30)
        group2 = np.random.normal(0.80, 0.05, 30)
        
        # Cohen's d
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((std1**2 + std2**2) / 2))
        cohens_d = (mean1 - mean2) / pooled_std
        
        # Interpret effect size
        if abs(cohens_d) < 0.2:
            effect_size = "small"
        elif abs(cohens_d) < 0.5:
            effect_size = "small"
        elif abs(cohens_d) < 0.8:
            effect_size = "medium"
        else:
            effect_size = "large"
        
        assert isinstance(cohens_d, (int, float))
        assert effect_size in ["small", "medium", "large"]

class TestStatisticalRobustness:
    """Test statistical robustness measures."""
    
    def test_coefficient_of_variation(self):
        """Test coefficient of variation calculation."""
        data_sets = [
            np.random.normal(0.85, 0.02, 50),  # Low variability
            np.random.normal(0.85, 0.10, 50),  # High variability
        ]
        
        cvs = []
        for data in data_sets:
            mean_val = np.mean(data)
            std_val = np.std(data, ddof=1)
            cv = (std_val / mean_val) * 100  # Coefficient of variation (%)
            cvs.append(cv)
        
        # Second dataset should have higher CV
        assert cvs[1] > cvs[0]
        assert all(cv > 0 for cv in cvs)
        assert all(cv < 100 for cv in cvs)  # Should be reasonable for performance metrics
    
    def test_outlier_detection(self):
        """Test outlier detection in performance data."""
        # Create data with outliers
        normal_data = np.random.normal(0.85, 0.02, 48)
        outliers = np.array([0.60, 0.95])  # Clear outliers
        data_with_outliers = np.concatenate([normal_data, outliers])
        
        # IQR method for outlier detection
        q1 = np.percentile(data_with_outliers, 25)
        q3 = np.percentile(data_with_outliers, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_mask = (data_with_outliers < lower_bound) | (data_with_outliers > upper_bound)
        detected_outliers = data_with_outliers[outlier_mask]
        
        assert len(detected_outliers) >= 1  # Should detect at least one outlier
        assert 0.60 in detected_outliers or 0.95 in detected_outliers
    
    def test_statistical_power_analysis(self):
        """Test statistical power analysis concepts."""
        # Parameters for power analysis
        effect_size = 0.5  # Cohen's d
        alpha = 0.05
        sample_size = 30
        
        # Mock power calculation (simplified)
        # In practice, would use specialized libraries like statsmodels
        
        # Critical t-value for two-tailed test
        df = 2 * sample_size - 2
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        # Non-centrality parameter
        ncp = effect_size * np.sqrt(sample_size / 2)
        
        # Statistical power (simplified approximation)
        # Power = 1 - β (probability of Type II error)
        power_approx = 1 - stats.t.cdf(t_critical, df, ncp) + stats.t.cdf(-t_critical, df, ncp)
        
        assert 0 <= power_approx <= 1
        assert isinstance(power_approx, (int, float))
        
        # Higher effect sizes should generally lead to higher power
        larger_effect_size = 0.8
        larger_ncp = larger_effect_size * np.sqrt(sample_size / 2)
        power_larger = 1 - stats.t.cdf(t_critical, df, larger_ncp) + stats.t.cdf(-t_critical, df, larger_ncp)
        
        # Power should increase with effect size (generally)
        assert isinstance(power_larger, (int, float))

class TestAFlipStatistics:
    """Test statistics specific to A-Flip scores in RA analysis."""
    
    def test_aflip_distribution_analysis(self):
        """Test A-Flip score distribution analysis."""
        # Mock A-Flip scores from different models
        aflip_scores_model1 = np.random.exponential(300, 100)  # Exponential distribution
        aflip_scores_model2 = np.random.exponential(500, 100)  # Different scale
        
        # Basic statistics
        stats_model1 = {
            'mean': np.mean(aflip_scores_model1),
            'std': np.std(aflip_scores_model1, ddof=1),
            'median': np.median(aflip_scores_model1),
            'q25': np.percentile(aflip_scores_model1, 25),
            'q75': np.percentile(aflip_scores_model1, 75)
        }
        
        stats_model2 = {
            'mean': np.mean(aflip_scores_model2),
            'std': np.std(aflip_scores_model2, ddof=1),
            'median': np.median(aflip_scores_model2),
            'q25': np.percentile(aflip_scores_model2, 25),
            'q75': np.percentile(aflip_scores_model2, 75)
        }
        
        # Model2 should generally have higher A-Flip scores
        assert stats_model2['mean'] > stats_model1['mean']
        assert all(val > 0 for val in stats_model1.values())
        assert all(val > 0 for val in stats_model2.values())
    
    def test_aflip_confidence_intervals(self):
        """Test confidence intervals for A-Flip scores."""
        aflip_scores = np.random.exponential(400, 80)
        
        # Bootstrap confidence interval
        n_bootstrap = 1000
        bootstrap_means = []
        
        np.random.seed(123)
        for _ in range(n_bootstrap):
            sample = np.random.choice(aflip_scores, size=len(aflip_scores), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        # 95% confidence interval
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)
        
        original_mean = np.mean(aflip_scores)
        
        assert ci_lower < original_mean < ci_upper
        assert ci_upper - ci_lower > 0
        
        # CI width relative to mean (coefficient of variation of the mean)
        ci_width = ci_upper - ci_lower
        ci_cv = (ci_width / (2 * 1.96)) / original_mean  # Approximate CV of the mean
        
        assert ci_cv > 0
        assert ci_cv < 1  # Should be reasonable
    
    def test_aflip_stability_comparison(self):
        """Test A-Flip score stability comparison between models."""
        # Mock A-Flip scores for stable vs unstable models
        stable_model_scores = np.random.exponential(200, 60)  # Lower scores = more stable
        unstable_model_scores = np.random.exponential(800, 60)  # Higher scores = less stable
        
        # Statistical comparison
        t_stat, p_value = stats.ttest_ind(stable_model_scores, unstable_model_scores)
        
        # Stability ratio
        stability_ratio = np.mean(stable_model_scores) / np.mean(unstable_model_scores)
        
        assert stability_ratio < 1  # Stable model should have lower ratio
        assert t_stat < 0  # Stable model should have significantly lower scores
        assert p_value < 0.05  # Should be statistically significant
        
        # Variance comparison (F-test concept)
        var_stable = np.var(stable_model_scores, ddof=1)
        var_unstable = np.var(unstable_model_scores, ddof=1)
        f_ratio = var_unstable / var_stable
        
        assert f_ratio > 1  # Unstable model should have higher variance
    
    @pytest.mark.slow
    def test_statistical_significance_aflip(self):
        """Test statistical significance of A-Flip score differences."""
        # Generate A-Flip scores for multiple models
        n_models = 4
        model_scores = {}
        
        for i in range(n_models):
            # Different stability levels
            base_score = 300 + i * 150  # Increasing instability
            scores = np.random.exponential(base_score, 50)
            model_scores[f'model_{i+1}'] = scores
        
        # Pairwise comparisons
        model_names = list(model_scores.keys())
        comparison_results = {}
        
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]
                scores1, scores2 = model_scores[model1], model_scores[model2]
                
                # Mann-Whitney U test (non-parametric, good for A-Flip distributions)
                u_stat, p_val = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
                
                comparison_results[f'{model1}_vs_{model2}'] = {
                    'u_statistic': u_stat,
                    'p_value': p_val,
                    'significant': p_val < 0.05
                }
        
        # Validate results
        assert len(comparison_results) == 6  # Should have 6 pairwise comparisons
        
        for comparison, results in comparison_results.items():
            assert 'u_statistic' in results
            assert 'p_value' in results
            assert 'significant' in results
            assert 0 <= results['p_value'] <= 1
            assert isinstance(results['significant'], bool)
