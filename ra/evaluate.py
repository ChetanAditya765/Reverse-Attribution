"""
Comprehensive evaluation framework for Reverse Attribution experiments.
Now properly integrated with your actual model implementations:
- BERTSentimentClassifier for text tasks
- ResNetCIFAR for vision tasks  
- Custom model examples for demonstration

Includes model evaluation, RA-specific metrics, and comparison with baseline methods.
Computes the 4 main metrics from the JMLR paper.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import json
from tqdm import tqdm
import logging
from pathlib import Path
import sys

# Add models directory to path for importing your actual models
models_path = Path(__file__).parent.parent / "models"
sys.path.insert(0, str(models_path))

# Import your actual model implementations
try:
    from models.bert_sentiment import BERTSentimentClassifier, BERTSentimentTrainer
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("⚠️ BERTSentimentClassifier not available for evaluation")

try:
    from models.resnet_cifar import (
        ResNetCIFAR, resnet56_cifar, resnet20_cifar, resnet32_cifar,
        get_model_info, ResNetCIFARTrainer
    )
    RESNET_AVAILABLE = True
except ImportError:
    RESNET_AVAILABLE = False
    print("⚠️ ResNet CIFAR models not available for evaluation")

try:
    from models.custom_model_example import (
        CustomTextClassifier, CustomVisionClassifier, CustomModelWrapper
    )
    CUSTOM_AVAILABLE = True
except ImportError:
    CUSTOM_AVAILABLE = False
    print("⚠️ Custom model examples not available for evaluation")

# Import RA framework components
from .ra import ReverseAttribution
from .metrics import (
    expected_calibration_error, compute_brier_score, jaccard_index,
    f1_score_localization, trust_change, evaluate_all_jmlr_metrics
)


class ModelEvaluator:
    """
    Main evaluation class that works with your actual model implementations.
    Provides comprehensive assessment including standard metrics, calibration,
    and Reverse Attribution analysis with proper model detection.
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
        
        # Detect your actual model type
        self.model_type = self._detect_model_type()
        
        # Initialize RA explainer with proper model detection
        self.ra_explainer = ReverseAttribution(self.model, device=self.device)
        
        # Setup logging
        self._setup_logging()
        
        print(f"✅ ModelEvaluator initialized with {self.model_type}")
        print(f"📱 Using device: {self.device}")
    
    def _detect_model_type(self) -> str:
        """Detect which of your actual models is being used."""
        model_class_name = self.model.__class__.__name__
        
        if model_class_name == "BERTSentimentClassifier":
            return "bert_sentiment"
        elif model_class_name == "ResNetCIFAR":
            return "resnet_cifar"
        elif model_class_name == "CustomTextClassifier":
            return "custom_text"
        elif model_class_name == "CustomVisionClassifier":
            return "custom_vision"
        elif hasattr(self.model, 'bert') and hasattr(self.model.bert, 'embeddings'):
            return "huggingface_bert"
        elif any(isinstance(module, torch.nn.Conv2d) for module in self.model.modules()):
            return "vision_cnn"
        else:
            return "unknown"
    
    def _setup_logging(self):
        """Setup logging for evaluation process."""
        log_path = os.path.join(self.save_dir, "evaluation.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def evaluate_standard_metrics(
        self,
        dataloader: torch.utils.data.DataLoader,
        dataset_name: str = "test"
    ) -> Dict[str, float]:
        """
        Compute standard classification metrics with proper handling of your models.
        """
        self.logger.info(f"🔍 Computing standard metrics on {dataset_name} set...")
        self.logger.info(f"📱 Model type detected: {self.model_type}")
        
        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0.0
        
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Standard Evaluation")):
                
                # Handle different batch formats based on your models
                if self.model_type in ["bert_sentiment", "custom_text", "huggingface_bert"]:
                    # Your text models
                    if isinstance(batch, dict):
                        inputs = batch['input_ids'].to(self.device)
                        attention_mask = batch.get('attention_mask', None)
                        if attention_mask is not None:
                            attention_mask = attention_mask.to(self.device)
                        labels = batch['labels'].to(self.device)
                        
                        # Forward pass with your BERT model
                        if attention_mask is not None:
                            outputs = self.model(inputs, attention_mask)
                        else:
                            outputs = self.model(inputs)
                    else:
                        # Handle other text formats
                        inputs, labels = batch
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs)
                
                elif self.model_type in ["resnet_cifar", "custom_vision", "vision_cnn"]:
                    # Your vision models
                    if isinstance(batch, dict):
                        inputs = batch['images'].to(self.device)
                        labels = batch['labels'].to(self.device)
                    else:
                        inputs, labels = batch
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    # Forward pass with your ResNet model
                    outputs = self.model(inputs)
                
                else:
                    # Fallback for unknown model types
                    if isinstance(batch, dict):
                        inputs = list(batch.values())[0].to(self.device)
                        labels = batch.get('labels', list(batch.values())[1]).to(self.device)
                    else:
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
        
        # Compute standard metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
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
            'num_samples': len(all_labels),
            'model_type': self.model_type,
            'model_class': self.model.__class__.__name__
        }
        
        # Add model-specific information
        if self.model_type == "resnet_cifar" and RESNET_AVAILABLE:
            try:
                model_info = get_model_info(self.model)
                metrics.update({
                    'total_parameters': model_info['total_parameters'],
                    'model_size_mb': model_info['model_size_mb']
                })
            except:
                pass
        
        elif self.model_type == "bert_sentiment" and BERT_AVAILABLE:
            try:
                if hasattr(self.model, 'get_model_info'):
                    model_info = self.model.get_model_info()
                    metrics.update({
                        'total_parameters': model_info['total_parameters'],
                        'vocab_size': model_info['vocab_size'],
                        'hidden_size': model_info['hidden_size']
                    })
            except:
                pass
        
        # Save detailed results
        results = {
            'predictions': all_preds.tolist(),
            'labels': all_labels.tolist(),
            'probabilities': all_probs.tolist(),
            'metrics': metrics
        }
        
        results_path = os.path.join(self.save_dir, f'{dataset_name}_standard_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"📊 Standard metrics computed - Accuracy: {accuracy:.4f}, ECE: {ece:.4f}")
        return metrics
    
    def evaluate_reverse_attribution(
        self,
        dataloader: torch.utils.data.DataLoader,
        dataset_name: str = "test",
        max_samples: int = 500,
        focus_on_errors: bool = True,
        top_m: int = 10
    ) -> Dict[str, Any]:
        """
        Comprehensive RA evaluation that works with your actual models.
        """
        self.logger.info(f"🔬 Running Reverse Attribution analysis on {dataset_name} set...")
        self.logger.info(f"🤖 Model type: {self.model_type}")
        
        ra_results = {
            'a_flip_scores': [],
            'counter_evidence_counts': [],
            'counter_evidence_strengths': [],
            'sample_indices': [],
            'predictions': [],
            'true_labels': [],
            'is_correct': [],
            'detailed_explanations': []
        }
        
        # Identify target samples (errors first if requested)
        if focus_on_errors:
            target_indices = self._identify_error_samples(dataloader, max_samples)
        else:
            target_indices = list(range(min(max_samples, len(dataloader.dataset))))
        
        self.logger.info(f"🎯 Analyzing {len(target_indices)} samples...")
        
        # RA analysis on selected samples
        analyzed_count = 0
        sample_idx = 0
        
        for batch in tqdm(dataloader, desc="RA Analysis"):
            if analyzed_count >= max_samples:
                break
            
            # Determine batch size based on model type
            if self.model_type in ["bert_sentiment", "custom_text", "huggingface_bert"]:
                if isinstance(batch, dict):
                    batch_size = len(batch['input_ids'])
                else:
                    batch_size = len(batch[0])
            else:
                if isinstance(batch, dict):
                    batch_size = len(list(batch.values())[0])
                else:
                    batch_size = len(batch[0])
            
            for i in range(batch_size):
                if sample_idx in target_indices and analyzed_count < max_samples:
                    
                    # Extract single sample based on model type
                    if self.model_type in ["bert_sentiment", "custom_text", "huggingface_bert"]:
                        if isinstance(batch, dict):
                            sample_input = batch['input_ids'][i:i+1].to(self.device)
                            sample_attention = batch.get('attention_mask', None)
                            if sample_attention is not None:
                                sample_attention = sample_attention[i:i+1].to(self.device)
                            true_label = batch['labels'][i].item()
                            
                            # RA analysis with attention mask as additional forward args
                            additional_args = (sample_attention,) if sample_attention is not None else None
                        else:
                            sample_input = batch[0][i:i+1].to(self.device)
                            true_label = batch[1][i].item()
                            additional_args = None
                    
                    else:  # Vision models
                        if isinstance(batch, dict):
                            sample_input = batch['images'][i:i+1].to(self.device)
                            true_label = batch['labels'][i].item()
                        else:
                            sample_input = batch[0][i:i+1].to(self.device)
                            true_label = batch[1][i].item()
                        additional_args = None
                    
                    try:
                        # Run RA analysis with your model
                        ra_result = self.ra_explainer.explain(
                            sample_input,
                            y_true=true_label,
                            top_m=top_m,
                            additional_forward_args=additional_args
                        )
                        
                        # Store results
                        ra_results['a_flip_scores'].append(ra_result['a_flip'])
                        ra_results['counter_evidence_counts'].append(len(ra_result['counter_evidence']))
                        
                        # Compute counter-evidence strength
                        if ra_result['counter_evidence']:
                            avg_strength = np.mean([abs(ce[2]) for ce in ra_result['counter_evidence']])
                            ra_results['counter_evidence_strengths'].append(avg_strength)
                        else:
                            ra_results['counter_evidence_strengths'].append(0.0)
                        
                        ra_results['sample_indices'].append(sample_idx)
                        ra_results['predictions'].append(ra_result['y_hat'])
                        ra_results['true_labels'].append(true_label)
                        ra_results['is_correct'].append(ra_result['y_hat'] == true_label)
                        
                        # Store detailed explanation for analysis
                        detailed_explanation = {
                            'sample_idx': sample_idx,
                            'a_flip': ra_result['a_flip'],
                            'counter_evidence': ra_result['counter_evidence'],
                            'prediction': ra_result['y_hat'],
                            'true_label': true_label,
                            'runner_up': ra_result['runner_up'],
                            'model_type': ra_result.get('model_type', self.model_type)
                        }
                        ra_results['detailed_explanations'].append(detailed_explanation)
                        
                        analyzed_count += 1
                        
                    except Exception as e:
                        self.logger.warning(f"⚠️ RA analysis failed for sample {sample_idx}: {e}")
                        continue
                
                sample_idx += 1
        
        # Compute summary statistics
        ra_summary = self._compute_ra_summary(ra_results)
        
        # Save RA results
        full_results = {
            'summary': ra_summary,
            'detailed_results': ra_results,
            'model_type': self.model_type,
            'model_class': self.model.__class__.__name__
        }
        
        results_path = os.path.join(self.save_dir, f'{dataset_name}_ra_results.json')
        with open(results_path, 'w') as f:
            json.dump(full_results, f, indent=2, default=str)
        
        self.logger.info(f"✅ RA analysis completed - {analyzed_count} samples analyzed")
        self.logger.info(f"📊 Avg A-Flip: {ra_summary['avg_a_flip']:.4f}, Avg CE: {ra_summary['avg_counter_evidence_count']:.2f}")
        
        return full_results
    
    def _identify_error_samples(self, dataloader: torch.utils.data.DataLoader, max_samples: int) -> List[int]:
        """Identify misclassified samples for focused RA analysis."""
        error_indices = []
        correct_indices = []
        sample_idx = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Identifying errors"):
                # Handle different batch formats
                if self.model_type in ["bert_sentiment", "custom_text", "huggingface_bert"]:
                    if isinstance(batch, dict):
                        inputs = batch['input_ids'].to(self.device)
                        attention_mask = batch.get('attention_mask', None)
                        if attention_mask is not None:
                            attention_mask = attention_mask.to(self.device)
                            outputs = self.model(inputs, attention_mask)
                        else:
                            outputs = self.model(inputs)
                        labels = batch['labels'].to(self.device)
                    else:
                        inputs, labels = batch
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs)
                else:
                    if isinstance(batch, dict):
                        inputs = batch['images'].to(self.device)
                        labels = batch['labels'].to(self.device)
                    else:
                        inputs, labels = batch
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                
                preds = torch.argmax(outputs, dim=1)
                
                for i in range(len(labels)):
                    if preds[i] != labels[i]:
                        error_indices.append(sample_idx)
                    else:
                        correct_indices.append(sample_idx)
                    sample_idx += 1
        
        # Select samples: prioritize errors, then add correct predictions
        target_indices = error_indices[:max_samples//2]
        remaining_slots = max_samples - len(target_indices)
        target_indices.extend(correct_indices[:remaining_slots])
        
        self.logger.info(f"🎯 Selected {len(error_indices)} errors and {len(correct_indices[:remaining_slots])} correct samples")
        return target_indices
    
    def _compute_ra_summary(self, ra_results: Dict[str, List]) -> Dict[str, Any]:
        """Compute summary statistics for RA analysis."""
        if not ra_results['a_flip_scores']:
            return {
                'avg_a_flip': 0.0,
                'std_a_flip': 0.0,
                'avg_counter_evidence_count': 0.0,
                'avg_counter_evidence_strength': 0.0,
                'samples_analyzed': 0,
                'error_samples': 0,
                'correct_samples': 0
            }
        
        summary = {
            'avg_a_flip': np.mean(ra_results['a_flip_scores']),
            'std_a_flip': np.std(ra_results['a_flip_scores']),
            'avg_counter_evidence_count': np.mean(ra_results['counter_evidence_counts']),
            'avg_counter_evidence_strength': np.mean(ra_results['counter_evidence_strengths']),
            'samples_analyzed': len(ra_results['a_flip_scores']),
            'error_samples': sum([not is_correct for is_correct in ra_results['is_correct']]),
            'correct_samples': sum(ra_results['is_correct']),
            'pct_samples_with_counter_evidence': np.mean([count > 0 for count in ra_results['counter_evidence_counts']]) * 100
        }
        
        # Additional statistics
        if ra_results['a_flip_scores']:
            summary.update({
                'max_a_flip': np.max(ra_results['a_flip_scores']),
                'min_a_flip': np.min(ra_results['a_flip_scores']),
                'median_a_flip': np.median(ra_results['a_flip_scores'])
            })
        
        return summary
    
    def evaluate_comprehensive(
        self,
        dataloader: torch.utils.data.DataLoader,
        dataset_name: str = "test",
        ra_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation combining standard and RA metrics.
        """
        self.logger.info(f"🚀 Starting comprehensive evaluation on {dataset_name}")
        
        # Default RA configuration
        if ra_config is None:
            ra_config = {
                'max_samples': 500,
                'focus_on_errors': True,
                'top_m': 10
            }
        
        # Standard evaluation
        standard_metrics = self.evaluate_standard_metrics(dataloader, dataset_name)
        
        # RA evaluation
        ra_analysis = self.evaluate_reverse_attribution(
            dataloader, 
            dataset_name,
            **ra_config
        )
        
        # Combined results
        comprehensive_results = {
            'dataset': dataset_name,
            'model_type': self.model_type,
            'model_class': self.model.__class__.__name__,
            'standard_metrics': standard_metrics,
            'ra_analysis': ra_analysis,
            'evaluation_config': ra_config
        }
        
        # Save comprehensive results
        results_path = os.path.join(self.save_dir, f'{dataset_name}_comprehensive_results.json')
        with open(results_path, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        self.logger.info(f"✅ Comprehensive evaluation completed and saved to {results_path}")
        return comprehensive_results
    
    def compare_with_baselines(
        self,
        dataloader: torch.utils.data.DataLoader,
        baseline_methods: List[str] = None,
        max_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Compare RA explanations with baseline explanation methods.
        """
        if baseline_methods is None:
            baseline_methods = ['integrated_gradients', 'shap']
        
        self.logger.info(f"🔄 Comparing RA with baselines: {baseline_methods}")
        
        comparison_results = {
            'ra_results': [],
            'baseline_results': {method: [] for method in baseline_methods},
            'comparison_metrics': {}
        }
        
        # This would implement comparison with baseline methods
        # For now, returning placeholder structure
        self.logger.warning("⚠️ Baseline comparison not fully implemented yet")
        
        return comparison_results


def evaluate_with_user_study_data(
    model: torch.nn.Module,
    user_study_results: Dict[str, List[float]],
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Evaluate using user study data for debugging time and trust metrics.
    Works with your actual model implementations.
    """
    metrics = {}
    
    # Debugging time analysis
    if 'debug_times_with_ra' in user_study_results and 'debug_times_without_ra' in user_study_results:
        with_ra_times = user_study_results['debug_times_with_ra']
        without_ra_times = user_study_results['debug_times_without_ra']
        
        metrics['avg_debug_time_with_ra'] = np.mean(with_ra_times)
        metrics['avg_debug_time_without_ra'] = np.mean(without_ra_times)
        metrics['debug_time_improvement'] = metrics['avg_debug_time_without_ra'] - metrics['avg_debug_time_with_ra']
        metrics['debug_time_improvement_pct'] = (metrics['debug_time_improvement'] / metrics['avg_debug_time_without_ra']) * 100
    
    # Trust change analysis
    if 'trust_before' in user_study_results and 'trust_after' in user_study_results:
        trust_before = user_study_results['trust_before']
        trust_after = user_study_results['trust_after']
        
        trust_changes = [after - before for before, after in zip(trust_before, trust_after)]
        metrics['avg_trust_change'] = np.mean(trust_changes)
        metrics['trust_improvement_rate'] = sum(1 for tc in trust_changes if tc > 0) / len(trust_changes)
        metrics['significant_trust_changes'] = sum(1 for tc in trust_changes if abs(tc) > 0.5) / len(trust_changes)
    
    # Model-specific metrics
    model_type = model.__class__.__name__
    metrics['model_type'] = model_type
    
    if model_type == "BERTSentimentClassifier":
        metrics['model_integration'] = "bert_sentiment"
    elif model_type == "ResNetCIFAR":
        metrics['model_integration'] = "resnet_cifar"
    else:
        metrics['model_integration'] = "custom_model"
    
    return metrics


def create_evaluation_report(
    evaluation_results: Dict[str, Any],
    output_path: str = "./evaluation_report.md"
) -> str:
    """
    Generate comprehensive evaluation report from results.
    """
    report_lines = [
        "# Reverse Attribution - Evaluation Report",
        f"Generated with actual model implementations\n",
        f"**Model Type**: {evaluation_results.get('model_type', 'Unknown')}",
        f"**Model Class**: {evaluation_results.get('model_class', 'Unknown')}",
        f"**Dataset**: {evaluation_results.get('dataset', 'Unknown')}\n",
        
        "## Standard Metrics",
        "| Metric | Value |",
        "|--------|-------|"
    ]
    
    if 'standard_metrics' in evaluation_results:
        metrics = evaluation_results['standard_metrics']
        report_lines.extend([
            f"| Accuracy | {metrics.get('accuracy', 0):.4f} |",
            f"| Precision | {metrics.get('precision', 0):.4f} |",
            f"| Recall | {metrics.get('recall', 0):.4f} |",
            f"| F1 Score | {metrics.get('f1', 0):.4f} |",
            f"| ECE | {metrics.get('ece', 0):.4f} |",
            f"| Brier Score | {metrics.get('brier_score', 0):.4f} |"
        ])
    
    report_lines.append("\n## Reverse Attribution Analysis")
    
    if 'ra_analysis' in evaluation_results and 'summary' in evaluation_results['ra_analysis']:
        ra_summary = evaluation_results['ra_analysis']['summary']
        report_lines.extend([
            "| RA Metric | Value |",
            "|-----------|-------|",
            f"| Avg A-Flip Score | {ra_summary.get('avg_a_flip', 0):.4f} |",
            f"| Std A-Flip Score | {ra_summary.get('std_a_flip', 0):.4f} |",
            f"| Avg Counter-Evidence Count | {ra_summary.get('avg_counter_evidence_count', 0):.2f} |",
            f"| Avg Counter-Evidence Strength | {ra_summary.get('avg_counter_evidence_strength', 0):.4f} |",
            f"| Samples Analyzed | {ra_summary.get('samples_analyzed', 0)} |",
            f"| Error Samples | {ra_summary.get('error_samples', 0)} |",
            f"| % with Counter-Evidence | {ra_summary.get('pct_samples_with_counter_evidence', 0):.1f}% |"
        ])
    
    # Write report
    report_content = '\n'.join(report_lines)
    with open(output_path, 'w') as f:
        f.write(report_content)
    
    return output_path


# Convenience functions for your model types
def evaluate_bert_sentiment_model(
    model: Any,  # Your BERTSentimentClassifier
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Convenience function for evaluating your BERT sentiment models."""
    evaluator = ModelEvaluator(model, device=device)
    return evaluator.evaluate_comprehensive(dataloader, "bert_sentiment_eval")


def evaluate_resnet_cifar_model(
    model: Any,  # Your ResNetCIFAR
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Convenience function for evaluating your ResNet CIFAR models."""
    evaluator = ModelEvaluator(model, device=device)
    return evaluator.evaluate_comprehensive(dataloader, "resnet_cifar_eval")


# Model availability check
def check_evaluation_compatibility():
    """Check which model types are available for evaluation."""
    compatibility = {
        'bert_sentiment': BERT_AVAILABLE,
        'resnet_cifar': RESNET_AVAILABLE,
        'custom_models': CUSTOM_AVAILABLE
    }
    
    print("📋 Evaluation Compatibility Status:")
    for model_type, available in compatibility.items():
        status = "✅" if available else "❌"
        print(f"  {status} {model_type}")
    
    return compatibility


if __name__ == "__main__":
    # Quick compatibility check
    check_evaluation_compatibility()
    
    # Example usage with your models
    if BERT_AVAILABLE:
        print("\n🧪 Testing BERT sentiment evaluation...")
        from models.bert_sentiment import BERTSentimentClassifier
        
        # Create test model
        test_bert = BERTSentimentClassifier("bert-base-uncased", num_classes=2)
        evaluator = ModelEvaluator(test_bert)
        print(f"✅ BERT evaluator created - detected type: {evaluator.model_type}")
    
    if RESNET_AVAILABLE:
        print("\n🧪 Testing ResNet CIFAR evaluation...")
        from models.resnet_cifar import resnet56_cifar
        
        # Create test model
        test_resnet = resnet56_cifar(num_classes=10)
        evaluator = ModelEvaluator(test_resnet)
        print(f"✅ ResNet evaluator created - detected type: {evaluator.model_type}")
    
    print("\n🎉 Evaluation framework integration test completed!")
