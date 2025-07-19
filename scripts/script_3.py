"""
Comprehensive evaluation script that properly integrates with your actual models.
Now uses BERTSentimentClassifier, ResNetCIFAR, and custom model implementations.
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import os
import json
from tqdm import tqdm
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import your actual model implementations
from ra.models.bert_sentiment import BERTSentimentClassifier, BERTSentimentTrainer
from ra.models.resnet_cifar import (
    ResNetCIFAR, resnet56_cifar, resnet20_cifar, resnet32_cifar, 
    ResNetCIFARTrainer, get_model_info
)
from ra.models.custom_model_example import CustomTextClassifier, CustomVisionClassifier

# Import RA framework
from ra.ra import ReverseAttribution
from ra.dataset_utils import DatasetLoader
from ra.evaluate import ModelEvaluator
from ra.metrics import expected_calibration_error, compute_brier_score


def evaluate_text_model(model_path: str, dataset_name: str, config: dict):
    """Evaluate your actual BERT sentiment model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load your actual BERT sentiment model
    model = BERTSentimentClassifier(
        model_name=config['model_name'],
        num_classes=config['num_classes']
    )
    
    # Load checkpoint
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✅ Loaded checkpoint from: {model_path}")
    else:
        print(f"⚠️ Checkpoint not found: {model_path}")
        return None
    
    model.to(device).eval()
    
    # Load data
    loader = DatasetLoader()
    test_dataloader = loader.create_text_dataloader(
        dataset_name=dataset_name,
        split="test",
        tokenizer=model.tokenizer,
        batch_size=32,
        shuffle=False
    )
    
    # Initialize RA with your model
    ra = ReverseAttribution(model, device=device)
    
    # Standard evaluation
    print(f"🔍 Evaluating {dataset_name} model...")
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Standard Evaluation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass with your BERT model
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Compute standard metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(test_dataloader)
    confidences = np.max(all_probs, axis=1)
    correct_predictions = (all_preds == all_labels).astype(int)
    ece = expected_calibration_error(confidences, correct_predictions)
    brier = compute_brier_score(all_probs, all_labels)
    
    standard_metrics = {
        'accuracy': accuracy,
        'avg_loss': avg_loss,
        'ece': ece,
        'brier_score': brier,
        'num_samples': len(all_labels),
        'model_type': 'BERTSentimentClassifier'
    }
    
    # RA analysis on error samples
    print("🔬 Running RA analysis...")
    ra_results = []
    sample_count = 0
    max_ra_samples = 200
    
    for batch in tqdm(test_dataloader, desc="RA Analysis"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
        
        # Run RA on misclassified samples
        for i in range(len(preds)):
            if preds[i] != labels[i] and len(ra_results) < max_ra_samples:
                # Extract single sample
                single_input = input_ids[i:i+1]
                single_attention = attention_mask[i:i+1]
                true_label = labels[i].item()
                
                # Run RA with attention mask as additional forward args
                ra_result = ra.explain(
                    single_input,
                    y_true=true_label,
                    additional_forward_args=(single_attention,)
                )
                ra_results.append(ra_result)
    
    # Compute RA summary statistics
    if ra_results:
        a_flip_scores = [r['a_flip'] for r in ra_results]
        counter_evidence_counts = [len(r['counter_evidence']) for r in ra_results]
        counter_evidence_strengths = []
        
        for r in ra_results:
            if r['counter_evidence']:
                avg_strength = np.mean([abs(ce[2]) for ce in r['counter_evidence']])
                counter_evidence_strengths.append(avg_strength)
            else:
                counter_evidence_strengths.append(0.0)
        
        ra_summary = {
            'avg_a_flip': np.mean(a_flip_scores),
            'std_a_flip': np.std(a_flip_scores),
            'avg_counter_evidence_count': np.mean(counter_evidence_counts),
            'avg_counter_evidence_strength': np.mean(counter_evidence_strengths),
            'samples_analyzed': len(ra_results),
            'model_types_detected': list(set([r.get('model_type', 'unknown') for r in ra_results]))
        }
    else:
        ra_summary = {
            'avg_a_flip': 0.0,
            'std_a_flip': 0.0,
            'avg_counter_evidence_count': 0.0,
            'avg_counter_evidence_strength': 0.0,
            'samples_analyzed': 0,
            'model_types_detected': []
        }
    
    print(f"\n{dataset_name.upper()} Results:")
    print(f"Model Type: {standard_metrics['model_type']}")
    print(f"Accuracy: {standard_metrics['accuracy']:.4f}")
    print(f"ECE: {standard_metrics['ece']:.4f}")
    print(f"Brier Score: {standard_metrics['brier_score']:.4f}")
    print(f"RA A-Flip Score (avg): {ra_summary['avg_a_flip']:.4f}")
    print(f"Counter-evidence features found: {ra_summary['avg_counter_evidence_count']:.2f}")
    print(f"Model types detected: {ra_summary['model_types_detected']}")
    
    return {
        'standard_metrics': standard_metrics,
        'ra_analysis': {'summary': ra_summary, 'detailed_results': ra_results},
        'dataset': dataset_name
    }


def evaluate_vision_model(model_path: str, config: dict):
    """Evaluate your actual ResNet CIFAR model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create your actual ResNet model
    architecture = config.get('architecture', 'resnet56')
    if architecture == 'resnet56':
        model = resnet56_cifar(num_classes=config['num_classes'])
    elif architecture == 'resnet20':
        model = resnet20_cifar(num_classes=config['num_classes'])
    elif architecture == 'resnet32':
        model = resnet32_cifar(num_classes=config['num_classes'])
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
    # Load checkpoint
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✅ Loaded checkpoint from: {model_path}")
    else:
        print(f"⚠️ Checkpoint not found: {model_path}")
        return None
    
    model.to(device).eval()
    
    # Load data
    loader = DatasetLoader()
    test_dataloader = loader.create_vision_dataloader(
        split="test",
        batch_size=128,
        shuffle=False
    )
    
    # Initialize RA with your model
    ra = ReverseAttribution(model, device=device)
    
    # Standard evaluation
    print("🔍 Evaluating CIFAR-10 model...")
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_dataloader, desc="Standard Evaluation")):
            data, target = data.to(device), target.to(device)
            
            # Forward pass with your ResNet model
            logits = model(data)
            loss = criterion(logits, target)
            total_loss += loss.item()
            
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Compute standard metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(test_dataloader)
    confidences = np.max(all_probs, axis=1)
    correct_predictions = (all_preds == all_labels).astype(int)
    ece = expected_calibration_error(confidences, correct_predictions)
    brier = compute_brier_score(all_probs, all_labels)
    
    # Get model info
    model_info = get_model_info(model)
    
    standard_metrics = {
        'accuracy': accuracy,
        'avg_loss': avg_loss,
        'ece': ece,
        'brier_score': brier,
        'num_samples': len(all_labels),
        'model_type': 'ResNetCIFAR',
        'architecture': architecture,
        'total_parameters': model_info['total_parameters']
    }
    
    # RA analysis on error samples
    print("🔬 Running RA analysis...")
    ra_results = []
    max_ra_samples = 200
    
    for batch_idx, (data, target) in enumerate(tqdm(test_dataloader, desc="RA Analysis")):
        if len(ra_results) >= max_ra_samples:
            break
            
        data, target = data.to(device), target.to(device)
        
        with torch.no_grad():
            logits = model(data)
            preds = torch.argmax(logits, dim=1)
        
        # Run RA on misclassified samples
        for i in range(len(preds)):
            if preds[i] != target[i] and len(ra_results) < max_ra_samples:
                single_input = data[i:i+1]
                true_label = target[i].item()
                
                ra_result = ra.explain(single_input, y_true=true_label)
                ra_results.append(ra_result)
    
    # Compute RA summary
    if ra_results:
        a_flip_scores = [r['a_flip'] for r in ra_results]
        counter_evidence_counts = [len(r['counter_evidence']) for r in ra_results]
        
        ra_summary = {
            'avg_a_flip': np.mean(a_flip_scores),
            'std_a_flip': np.std(a_flip_scores),
            'avg_counter_evidence_count': np.mean(counter_evidence_counts),
            'samples_analyzed': len(ra_results),
            'model_types_detected': list(set([r.get('model_type', 'unknown') for r in ra_results]))
        }
    else:
        ra_summary = {
            'avg_a_flip': 0.0,
            'std_a_flip': 0.0,
            'avg_counter_evidence_count': 0.0,
            'samples_analyzed': 0,
            'model_types_detected': []
        }
    
    print(f"\nCIFAR-10 Results:")
    print(f"Model Type: {standard_metrics['model_type']}")
    print(f"Architecture: {standard_metrics['architecture']}")
    print(f"Parameters: {standard_metrics['total_parameters']:,}")
    print(f"Accuracy: {standard_metrics['accuracy']:.4f}")
    print(f"ECE: {standard_metrics['ece']:.4f}")
    print(f"RA A-Flip Score (avg): {ra_summary['avg_a_flip']:.4f}")
    print(f"Model types detected: {ra_summary['model_types_detected']}")
    
    return {
        'standard_metrics': standard_metrics,
        'ra_analysis': {'summary': ra_summary, 'detailed_results': ra_results},
        'dataset': 'cifar10'
    }


def evaluate_all_models(config: dict):
    """Evaluate all trained models using your actual model implementations."""
    results = {}
    
    print("🔍 Starting comprehensive model evaluation with your actual models...")
    print("=" * 70)
    
    # Evaluate text models (your BERT implementations)
    if 'text_models' in config:
        print("\n📚 Evaluating your BERT sentiment models...")
        
        for dataset_name, model_config in config['text_models'].items():
            model_path = os.path.join(model_config['output_dir'], 'best_model.pt')
            
            print(f"\n  📖 Evaluating {dataset_name} model...")
            print(f"     Model: {model_config['model_name']}")
            print(f"     Classes: {model_config['num_classes']}")
            
            result = evaluate_text_model(model_path, dataset_name, model_config)
            if result:
                results[f'{dataset_name}_results'] = result
            else:
                print(f"  ❌ Failed to evaluate {dataset_name} model")
    
    # Evaluate vision models (your ResNet implementations)
    if 'vision_models' in config:
        print("\n🖼️ Evaluating your ResNet CIFAR models...")
        
        for dataset_name, model_config in config['vision_models'].items():
            model_path = os.path.join(model_config['output_dir'], 'best_model.pt')
            
            print(f"\n  🏞️ Evaluating {dataset_name} model...")
            print(f"     Architecture: {model_config['architecture']}")
            print(f"     Classes: {model_config['num_classes']}")
            
            result = evaluate_vision_model(model_path, model_config)
            if result:
                results[f'{dataset_name}_results'] = result
            else:
                print(f"  ❌ Failed to evaluate {dataset_name} model")
    
    # Save comprehensive results
    results_path = 'comprehensive_evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Comprehensive evaluation completed!")
    print(f"📊 Results saved to: {results_path}")
    print("=" * 70)
    
    # Print summary table
    print("\n📋 EVALUATION SUMMARY")
    print("-" * 50)
    
    for key, result in results.items():
        if result and 'standard_metrics' in result:
            metrics = result['standard_metrics']
            ra_summary = result['ra_analysis']['summary']
            
            print(f"\n{key.replace('_results', '').upper()}:")
            print(f"  Model Type: {metrics.get('model_type', 'Unknown')}")
            if 'architecture' in metrics:
                print(f"  Architecture: {metrics['architecture']}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  ECE: {metrics['ece']:.4f}")
            print(f"  A-Flip: {ra_summary['avg_a_flip']:.4f}")
            print(f"  RA Samples: {ra_summary['samples_analyzed']}")
            print(f"  Detected Types: {ra_summary['model_types_detected']}")
    
    return results


if __name__ == "__main__":
    # Example usage with default config
    import yaml
    
    # Create default config if needed
    default_config = {
        'text_models': {
            'imdb': {
                'model_name': 'bert-base-uncased',
                'num_classes': 2,
                'output_dir': './checkpoints/bert_imdb'
            },
            'yelp': {
                'model_name': 'roberta-large',
                'num_classes': 2,
                'output_dir': './checkpoints/roberta_yelp'
            }
        },
        'vision_models': {
            'cifar10': {
                'architecture': 'resnet56',
                'num_classes': 10,
                'output_dir': './checkpoints/resnet56_cifar10'
            }
        }
    }
    
    results = evaluate_all_models(default_config)
    print("\n🎉 Evaluation completed with your actual model implementations!")
