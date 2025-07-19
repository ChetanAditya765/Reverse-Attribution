"""
Comprehensive evaluation framework for Reverse Attribution experiments.
Includes model evaluation, RA-specific metrics, and comparison with baseline methods.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from ra.ra import ReverseAttribution
from ra.model_factory import ModelFactory
from ra.dataset_utils import DatasetLoader
from metrics import (
    expected_calibration_error, jaccard_index, f1_score_localization,
    average_debug_time, trust_change, compute_brier_score
)
from explainer_utils import ExplainerHub


class ModelEvaluator:
    """
    Main evaluation class for comprehensive model assessment including
    standard metrics, calibration, and Reverse Attribution analysis.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = None,
        save_dir: str = "./evaluation_results"
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize RA explainer
        self.ra_explainer = ReverseAttribution(self.model, device=self.device)
        
        # Initialize baseline explainers
        self.baseline_explainers = ExplainerHub(self.model)
    
    def evaluate_standard_metrics(
        self,
        dataloader: torch.utils.data.DataLoader,
        dataset_name: str = "test"
    ) -> Dict[str, float]:
        """
        Compute standard classification metrics: accuracy, precision, recall, F1.
        
        Args:
            dataloader: DataLoader for evaluation dataset
            dataset_name: Name of dataset for logging
            
        Returns:
            Dictionary of computed metrics
        """
        print(f"🔍 Computing standard metrics on {dataset_name} set...")
        
        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0.0
        
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Standard Evaluation")):
                
                # Handle different batch formats (text vs vision)
                if isinstance(batch, dict):  # Text data
                    inputs = batch['input_ids'].to(self.device)
                    attention_mask = batch.get('attention_mask', None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    if attention_mask is not None:
                        outputs = self.model(inputs, attention_mask)
                    else:
                        outputs = self.model(inputs)
                else:  # Vision data
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro'
        )
        avg_loss = total_loss / len(dataloader)
        
        # Calibration metrics
        confidences = np.max(all_probs, axis=1)
        correct_predictions = (all_preds == all_labels).astype(int)
        ece = expected_calibration_error(confidences, correct_predictions)
        brier = compute_brier_score(all_probs, all_labels)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_loss': avg_loss,
            'ece': ece,
            'brier_score': brier,
            'num_samples': len(all_labels)
        }
        
        # Save predictions and probabilities
        results = {
            'predictions': all_preds.tolist(),
            'labels': all_labels.tolist(),
            'probabilities': all_probs.tolist(),
            'metrics': metrics
        }
        
        results_path = os.path.join(self.save_dir, f'{dataset_name}_standard_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return metrics
    
    def evaluate_reverse_attribution(
        self,
        dataloader: torch.utils.data.DataLoader,
        dataset_name: str = "test",
        max_samples: int = 500,
        focus_on_errors: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate Reverse Attribution on dataset samples.
        
        Args:
            dataloader: DataLoader for evaluation
            dataset_name: Name of dataset for logging
            max_samples: Maximum number of samples to evaluate
            focus_on_errors: Whether to prioritize misclassified examples
            
        Returns:
            Dictionary containing RA evaluation results
        """
        print(f"🔬 Running Reverse Attribution analysis on {dataset_name} set...")
        
        ra_results = {
            'a_flip_scores': [],
            'counter_evidence_counts': [],
            'counter_evidence_strengths': [],
            'sample_indices': [],
            'predictions': [],
            'true_labels': [],
            'is_correct': []
        }
        
        # First pass: get all predictions to identify errors if needed
        if focus_on_errors:
            print("  First pass: identifying misclassified examples...")
            all_indices = []
            error_indices = []
            
            with torch.no_grad():
                sample_idx = 0
                for batch in tqdm(dataloader, desc="Finding errors"):
                    if isinstance(batch, dict):  # Text data
                        inputs = batch['input_ids'].to(self.device)
                        attention_mask = batch.get('attention_mask', None)
                        if attention_mask is not None:
                            attention_mask = attention_mask.to(self.device)
                        labels = batch['labels'].to(self.device)
                        
                        if attention_mask is not None:
                            outputs = self.model(inputs, attention_mask)
                        else:
                            outputs = self.model(inputs)
                    else:  # Vision data
                        inputs, labels = batch
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs)
                    
                    preds = torch.argmax(outputs, dim=1)
                    
                    for i in range(len(labels)):
                        if preds[i] != labels[i]:
                            error_indices.append(sample_idx)
                        all_indices.append(sample_idx)
                        sample_idx += 1
            
            # Select samples: prioritize errors, then add correct predictions
            target_indices = error_indices[:max_samples//2]
            remaining_slots = max_samples - len(target_indices)
            correct_indices = [idx for idx in all_indices if idx not in error_indices]
            target_indices.extend(correct_indices[:remaining_slots])
        else:
            target_indices = list(range(min(max_samples, len(dataloader.dataset))))
        
        print(f"  Analyzing {len(target_indices)} samples...")
        
        # Second pass: RA analysis on selected samples
        sample_idx = 0
        analyzed_count = 0
        
        for batch in tqdm(dataloader, desc="RA Analysis"):
            if analyzed_count >= max_samples:
                break
                
            batch_size = len(batch['labels']) if isinstance(batch, dict) else len(batch[1])
            
            for i in range(batch_size):
                if sample_idx in target_indices and analyzed_count < max_samples:
                    # Extract single sample
                    if isinstance(batch, dict):  # Text data
                        sample_input = batch['input_ids'][i:i+1].to(self.device)
                        sample_attention = batch.get('attention_mask', None)
                        if sample_attention is not None:
                            sample_attention = sample_attention[i:i+1].to(self.device)
                        true_label = batch['labels'][i].item()
                        
                        # Run RA analysis
                        ra_result = self.ra_explainer.explain(
                            sample_input, y_true=true_label
                        )
                    else:  # Vision data
                        sample_input = batch[0][i:i+1].to(self.device)
                        true_label = batch[1][i].item()
                        
                        # Run RA analysis
                        ra_result = self.ra_explainer.explain(
                            sample_input, y_true=true_label
                        )
                    
                    # Store results
                    ra_results['a_flip_scores'].append(ra_result['a_flip'])
                    ra_results['counter_evidence_counts'].append(len(ra_result['counter_evidence']))
                    
                    # Compute average counter-evidence strength
                    if ra_result['counter_evidence']:
                        avg_strength = np.mean([abs(ce[2]) for ce in ra_result['counter_evidence']])
                        ra_results['counter_evidence_strengths'].append(avg_strength)
                    else:
                        ra_results['counter_evidence_strengths'].append(0.0)
                    
                    ra_results['sample_indices'].append(sample_idx)
                    ra_results['predictions'].append(ra_result['y_hat'])
                    ra_results['true_labels'].append(true_label)
                    ra_results['is_correct'].append(ra_result['y_hat'] == true_label)
                    
                    analyzed_count += 1
                
                sample_idx += 1
        
        # Compute summary statistics
        ra_summary = {
            'avg_a_flip': np.mean(ra_results['a_flip_scores']),
            'std_a_flip': np.std(ra_results['a_flip_scores']),
            'avg_counter_evidence_count': np.mean(ra_results['counter_evidence_counts']),
            'avg_counter_evidence_strength': np.mean(ra_results['counter_evidence_strengths']),
            'samples_analyzed': analyzed_count,
            'error_samples': sum([1 for is_correct in ra_results['is_correct'] if not is_correct])
        }
        
        # Save results
        full_results = {
            'summary': ra_summary,
            'detailed_results': ra_results
        }
        
        results_path = os.path.join(self.save_dir, f'{dataset_name}_ra_results.json')
        with open(results_path, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        return full_results
    
    def evaluate_misprediction_localization(
        self,
        dataloader: torch.utils.data.DataLoader,
        ground_truth_masks: Dict[int, List[int]],
        dataset_name: str = "test",
        top_k: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate how well RA localizes misprediction regions.
        
        Args:
            dataloader: DataLoader for evaluation
            ground_truth_masks: Dictionary mapping sample indices to ground truth regions
            dataset_name: Dataset name
            top_k: Number of top counter-evidence features to consider
            
        Returns:
            Localization metrics
        """
        print(f"🎯 Evaluating misprediction localization on {dataset_name} set...")
        
        jaccard_scores = []
        f1_scores = []
        
        sample_idx = 0
        
        for batch in tqdm(dataloader, desc="Localization Evaluation"):
            batch_size = len(batch['labels']) if isinstance(batch, dict) else len(batch[1])
            
            for i in range(batch_size):
                if sample_idx in ground_truth_masks:
                    # Extract sample
                    if isinstance(batch, dict):
                        sample_input = batch['input_ids'][i:i+1].to(self.device)
                        true_label = batch['labels'][i].item()
                    else:
                        sample_input = batch[0][i:i+1].to(self.device)
                        true_label = batch[1][i].item()
                    
                    # Get RA explanation
                    ra_result = self.ra_explainer.explain(sample_input, y_true=true_label)
                    
                    # Extract top-k counter-evidence indices
                    if ra_result['counter_evidence']:
                        predicted_indices = [ce[0] for ce in ra_result['counter_evidence'][:top_k]]
                    else:
                        predicted_indices = []
                    
                    # Get ground truth indices
                    true_indices = ground_truth_masks[sample_idx]
                    
                    # Compute metrics
                    jaccard = jaccard_index(set(predicted_indices), set(true_indices))
                    f1 = f1_score_localization(predicted_indices, true_indices)
                    
                    jaccard_scores.append(jaccard)
                    f1_scores.append(f1)
                
                sample_idx += 1
        
        localization_metrics = {
            'avg_jaccard': np.mean(jaccard_scores),
            'std_jaccard': np.std(jaccard_scores),
            'avg_f1': np.mean(f1_scores),
            'std_f1': np.std(f1_scores),
            'num_samples': len(jaccard_scores)
        }
        
        # Save results
        results_path = os.path.join(self.save_dir, f'{dataset_name}_localization_results.json')
        with open(results_path, 'w') as f:
            json.dump({
                'metrics': localization_metrics,
                'individual_scores': {
                    'jaccard_scores': jaccard_scores,
                    'f1_scores': f1_scores
                }
            }, f, indent=2)
        
        return localization_metrics
    
    def generate_evaluation_report(
        self,
        standard_metrics: Dict[str, float],
        ra_results: Dict[str, Any],
        localization_metrics: Dict[str, float] = None,
        dataset_name: str = "test"
    ) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            standard_metrics: Standard classification metrics
            ra_results: RA evaluation results
            localization_metrics: Localization evaluation results
            dataset_name: Dataset name for report
            
        Returns:
            Path to generated report
        """
        report_path = os.path.join(self.save_dir, f'{dataset_name}_evaluation_report.txt')
        
        with open(report_path, 'w') as f:
            f.write(f"REVERSE ATTRIBUTION EVALUATION REPORT\n")
            f.write(f"Dataset: {dataset_name.upper()}\n")
            f.write(f"{'='*60}\n\n")
            
            # Standard metrics
            f.write("STANDARD CLASSIFICATION METRICS\n")
            f.write("-" * 35 + "\n")
            f.write(f"Accuracy:     {standard_metrics['accuracy']:.4f}\n")
            f.write(f"Precision:    {standard_metrics['precision']:.4f}\n")
            f.write(f"Recall:       {standard_metrics['recall']:.4f}\n")
            f.write(f"F1-Score:     {standard_metrics['f1']:.4f}\n")
            f.write(f"Average Loss: {standard_metrics['avg_loss']:.4f}\n\n")
            
            # Calibration metrics
            f.write("CALIBRATION METRICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"ECE:          {standard_metrics['ece']:.4f}\n")
            f.write(f"Brier Score:  {standard_metrics['brier_score']:.4f}\n\n")
            
            # RA metrics
            f.write("REVERSE ATTRIBUTION ANALYSIS\n")
            f.write("-" * 30 + "\n")
            ra_summary = ra_results['summary']
            f.write(f"Samples Analyzed:           {ra_summary['samples_analyzed']}\n")
            f.write(f"Error Samples:              {ra_summary['error_samples']}\n")
            f.write(f"Avg A-Flip Score:           {ra_summary['avg_a_flip']:.4f} ± {ra_summary['std_a_flip']:.4f}\n")
            f.write(f"Avg Counter-Evidence Count: {ra_summary['avg_counter_evidence_count']:.2f}\n")
            f.write(f"Avg Counter-Evidence Strength: {ra_summary['avg_counter_evidence_strength']:.4f}\n\n")
            
            # Localization metrics if available
            if localization_metrics:
                f.write("MISPREDICTION LOCALIZATION\n")
                f.write("-" * 27 + "\n")
                f.write(f"Avg Jaccard Index: {localization_metrics['avg_jaccard']:.4f} ± {localization_metrics['std_jaccard']:.4f}\n")
                f.write(f"Avg F1 Score:      {localization_metrics['avg_f1']:.4f} ± {localization_metrics['std_f1']:.4f}\n\n")
            
            # Interpretation
            f.write("INTERPRETATION\n")
            f.write("-" * 15 + "\n")
            if ra_summary['avg_a_flip'] > 0.5:
                f.write("• High attribution instability detected - model predictions may be unreliable\n")
            else:
                f.write("• Low attribution instability - model predictions appear stable\n")
                
            if ra_summary['avg_counter_evidence_count'] > 2:
                f.write("• Significant counter-evidence found - consider investigating feature interactions\n")
            else:
                f.write("• Limited counter-evidence detected\n")
        
        print(f"📄 Evaluation report saved to: {report_path}")
        return report_path


def evaluate_with_user_study_data(
    model: torch.nn.Module,
    user_study_results: Dict[str, List[float]],
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Evaluate using user study data for debugging time and trust metrics.
    
    Args:
        model: PyTorch model
        user_study_results: Dictionary with user study results
        device: Device for computation
        
    Returns:
        User study evaluation metrics
    """
    metrics = {}
    
    # Debugging time analysis
    if 'debug_times_with_ra' in user_study_results and 'debug_times_without_ra' in user_study_results:
        with_ra_times = user_study_results['debug_times_with_ra']
        without_ra_times = user_study_results['debug_times_without_ra']
        
        metrics['avg_debug_time_with_ra'] = average_debug_time(with_ra_times)
        metrics['avg_debug_time_without_ra'] = average_debug_time(without_ra_times)
        metrics['debug_time_improvement'] = metrics['avg_debug_time_without_ra'] - metrics['avg_debug_time_with_ra']
    
    # Trust change analysis
    if 'trust_before' in user_study_results and 'trust_after' in user_study_results:
        trust_before = user_study_results['trust_before']
        trust_after = user_study_results['trust_after']
        
        metrics['avg_trust_change'] = trust_change(trust_before, trust_after)
        metrics['trust_improvement_rate'] = sum(1 for b, a in zip(trust_before, trust_after) if a > b) / len(trust_before)
    
    return metrics


if __name__ == "__main__":
    # Example usage
    from ra.model_factory import ModelFactory
    from ra.dataset_utils import DatasetLoader
    
    # Load model
    model = ModelFactory.create_text_model(
        model_name="bert-base-uncased",
        num_classes=2,
        checkpoint_path="./checkpoints/bert_imdb/best_model.pt"
    )
    
    # Create evaluator
    evaluator = ModelEvaluator(model, save_dir="./evaluation_results")
    
    # Load test data
    loader = DatasetLoader()
    test_dataloader = loader.create_text_dataloader(
        "imdb", "test", model.tokenizer, batch_size=32, shuffle=False
    )
    
    # Run comprehensive evaluation
    print("Starting comprehensive evaluation...")
    
    # Standard metrics
    standard_metrics = evaluator.evaluate_standard_metrics(test_dataloader, "imdb_test")
    print(f"Standard metrics: {standard_metrics}")
    
    # RA analysis
    ra_results = evaluator.evaluate_reverse_attribution(test_dataloader, "imdb_test", max_samples=200)
    print(f"RA analysis complete. Avg A-Flip: {ra_results['summary']['avg_a_flip']:.4f}")
    
    # Generate report
    report_path = evaluator.generate_evaluation_report(standard_metrics, ra_results, dataset_name="imdb_test")
    print(f"Evaluation report generated: {report_path}")
