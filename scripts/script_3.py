"""
Comprehensive evaluation script for all trained models.
Computes metrics reported in the paper including accuracy, ECE, and RA-specific metrics.
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import os
import json
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ra.model_factory import ModelFactory
from ra.dataset_utils import DatasetLoader
from ra.ra import ReverseAttribution
from ra.metrics import compute_ece, compute_trust_metrics
from ra.evaluate import evaluate_model_with_ra


def evaluate_text_model(model_path: str, dataset_name: str, config: dict):
    """Evaluate text classification model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = ModelFactory.create_text_model(
        model_name=config['model_name'],
        num_classes=config['num_classes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
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
    
    # Standard evaluation
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    ece = compute_ece(np.array(all_probs), np.array(all_labels))
    
    print(f"\n{dataset_name.upper()} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ECE: {ece:.4f}")
    
    # RA evaluation on subset of misclassified examples
    ra = ReverseAttribution(model, device=device)
    
    # Find misclassified examples
    misclassified_indices = [i for i, (pred, true) in enumerate(zip(all_preds, all_labels)) if pred != true]
    
    if len(misclassified_indices) > 0:
        # Evaluate RA on first 100 misclassified examples
        sample_indices = misclassified_indices[:100]
        
        ra_results = evaluate_model_with_ra(
            model, ra, test_dataloader, sample_indices, device
        )
        
        print(f"RA A-Flip Score (avg): {np.mean(ra_results['a_flip_scores']):.4f}")
        print(f"Counter-evidence features found: {len(ra_results['counter_evidence'])}")
    
    return {
        'accuracy': accuracy,
        'ece': ece,
        'num_misclassified': len(misclassified_indices),
        'total_samples': len(all_labels)
    }


def evaluate_vision_model(model_path: str, config: dict):
    """Evaluate vision model (ResNet-56 on CIFAR-10)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = ModelFactory.create_vision_model(
        num_classes=config['num_classes'],
        architecture=config['architecture']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    
    # Load data
    loader = DatasetLoader()
    test_dataloader = loader.create_vision_dataloader(
        split="test",
        batch_size=128,
        shuffle=False
    )
    
    # Standard evaluation
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in tqdm(test_dataloader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            probs = F.softmax(output, dim=1)
            pred = output.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    ece = compute_ece(np.array(all_probs), np.array(all_labels))
    
    print(f"\nCIFAR-10 Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ECE: {ece:.4f}")
    
    # RA evaluation
    ra = ReverseAttribution(model, device=device)
    
    # Find misclassified examples
    misclassified_indices = [i for i, (pred, true) in enumerate(zip(all_preds, all_labels)) if pred != true]
    
    if len(misclassified_indices) > 0:
        sample_indices = misclassified_indices[:100]
        
        ra_results = evaluate_model_with_ra(
            model, ra, test_dataloader, sample_indices, device
        )
        
        print(f"RA A-Flip Score (avg): {np.mean(ra_results['a_flip_scores']):.4f}")
    
    return {
        'accuracy': accuracy,
        'ece': ece,
        'num_misclassified': len(misclassified_indices),
        'total_samples': len(all_labels)
    }


def evaluate_all_models(config: dict):
    """Evaluate all trained models and generate comprehensive report."""
    results = {}
    
    print("🔍 Starting comprehensive model evaluation...")
    
    # Evaluate text models
    if 'text_models' in config:
        print("\n📚 Evaluating text models...")
        
        for dataset_name, model_config in config['text_models'].items():
            model_path = os.path.join(model_config['output_dir'], 'best_model.pt')
            
            if os.path.exists(model_path):
                print(f"\n  Evaluating {dataset_name} model...")
                results[f'{dataset_name}_results'] = evaluate_text_model(
                    model_path, dataset_name, model_config
                )
            else:
                print(f"  ⚠️  Model not found: {model_path}")
    
    # Evaluate vision models
    if 'vision_models' in config:
        print("\n🖼️ Evaluating vision models...")
        
        for dataset_name, model_config in config['vision_models'].items():
            model_path = os.path.join(model_config['output_dir'], 'best_model.pt')
            
            if os.path.exists(model_path):
                print(f"\n  Evaluating {dataset_name} model...")
                results[f'{dataset_name}_results'] = evaluate_vision_model(
                    model_path, model_config
                )
            else:
                print(f"  ⚠️  Model not found: {model_path}")
    
    # Save results
    results_path = 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Evaluation completed! Results saved to: {results_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  ECE: {result['ece']:.4f}")
        print(f"  Misclassified: {result['num_misclassified']}/{result['total_samples']}")


if __name__ == "__main__":
    # Example usage with default config
    config = {
        'text_models': {
            'imdb': {
                'model_name': 'bert-base-uncased',
                'num_classes': 2,
                'output_dir': './checkpoints/bert_imdb'
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
    
    evaluate_all_models(config)
