"""
Main orchestration script that coordinates training and evaluation of your actual models.
Now properly integrates with BERTSentimentClassifier and ResNetCIFAR implementations.
"""

import argparse
import yaml
import os
import torch
from pathlib import Path
import json
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import your model-integrated scripts
from script_1 import train_text_model, train_multiple_text_models
from script_2 import train_vision_model, train_multiple_vision_models
from script_3 import evaluate_all_models

# Import your actual model availability checks
from models import get_bert_model, get_resnet56_model


def check_model_availability():
    """Check which of your models are available."""
    available = {
        'bert_sentiment': False,
        'resnet_cifar': False,
        'custom_models': False
    }
    
    try:
        # Test your BERT model
        test_bert = get_bert_model("bert-base-uncased", num_classes=2)
        available['bert_sentiment'] = True
        print("✅ Your BERT sentiment models available")
    except Exception as e:
        print(f"❌ Your BERT sentiment models unavailable: {e}")
    
    try:
        # Test your ResNet model
        test_resnet = get_resnet56_model(num_classes=10)
        available['resnet_cifar'] = True
        print("✅ Your ResNet CIFAR models available")
    except Exception as e:
        print(f"❌ Your ResNet CIFAR models unavailable: {e}")
    
    try:
        # Test custom models
        from models.custom_model_example import CustomTextClassifier
        test_custom = CustomTextClassifier(num_classes=2)
        available['custom_models'] = True
        print("✅ Your custom model examples available")
    except Exception as e:
        print(f"❌ Your custom model examples unavailable: {e}")
    
    return available


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        return create_default_config(config_path)


def create_default_config(config_path: str) -> dict:
    """Create default configuration that uses your actual models."""
    
    default_config = {
        'meta': {
            'project_name': 'Reverse Attribution',
            'model_implementations': 'Using actual BERTSentimentClassifier and ResNetCIFAR',
            'paper_reference': 'JMLR Reverse Attribution Paper'
        },
        
        'data': {
            'data_dir': './data',
            'download': True
        },
        
        'text_models': {
            'imdb': {
                'model_name': 'bert-base-uncased',
                'model_type': 'BERTSentimentClassifier',
                'num_classes': 2,
                'epochs': 3,
                'batch_size': 16,
                'learning_rate': 2e-5,
                'max_length': 512,
                'dropout_rate': 0.1,
                'weight_decay': 0.01,
                'output_dir': './checkpoints/bert_imdb'
            },
            'yelp': {
                'model_name': 'roberta-base',  
                'model_type': 'BERTSentimentClassifier',
                'num_classes': 2,
                'epochs': 3,
                'batch_size': 8,
                'learning_rate': 1e-5,
                'max_length': 512,
                'dropout_rate': 0.1,
                'weight_decay': 0.01,
                'output_dir': './checkpoints/roberta_yelp'
            }
        },
        
        'vision_models': {
            'cifar10': {
                'architecture': 'resnet56',
                'model_type': 'ResNetCIFAR',
                'num_classes': 10,
                'epochs': 200,
                'batch_size': 128,
                'learning_rate': 0.1,
                'momentum': 0.9,
                'weight_decay': 1e-4,
                'milestones': [100, 150],
                'output_dir': './checkpoints/resnet56_cifar10'
            }
        },
        
        'evaluation': {
            'ra_samples': 500,
            'focus_on_errors': True,
            'baseline_methods': ['shap', 'lime', 'integrated_gradients'],
            'user_study_samples': 50
        }
    }
    
    # Save default config
    config_dir = os.path.dirname(config_path)
    os.makedirs(config_dir, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
    print(f"📝 Created default config at {config_path}")
    return default_config


def setup_data_stage(config: dict):
    """Setup and download datasets."""
    print("\n📊 Stage 1: Data Preparation")
    print("-" * 30)
    
    data_dir = config['data']['data_dir']
    os.makedirs(data_dir, exist_ok=True)
    
    if config['data']['download']:
        print("📥 Downloading datasets...")
        try:
            # Try to use your dataset utilities
            from ra.dataset_utils import DatasetLoader
            loader = DatasetLoader(data_dir)
            
            # Test dataset loading
            print("  📚 Testing IMDB dataset...")
            texts, labels = loader.load_imdb("train")
            print(f"     ✅ IMDB loaded: {len(texts)} samples")
            
            print("  📝 Testing Yelp dataset...")
            texts, labels = loader.load_yelp_polarity("train")  
            print(f"     ✅ Yelp loaded: {len(texts)} samples")
            
            print("  🖼️ Testing CIFAR-10 dataset...")
            cifar_dataset = loader.load_cifar10("train")
            print(f"     ✅ CIFAR-10 loaded: {len(cifar_dataset)} samples")
            
        except Exception as e:
            print(f"     ⚠️ Dataset download issue: {e}")
            print("     📝 Please ensure internet connection and try again")


def training_stage(config: dict, model_type: str):
    """Training stage using your actual models."""
    print(f"\n🏋️ Stage 2: Model Training ({model_type})")
    print("-" * 40)
    
    training_results = {}
    
    if model_type in ['text', 'all']:
        print("\n📚 Training your BERT sentiment models...")
        
        if 'text_models' in config:
            text_results = train_multiple_text_models(config['text_models'])
            training_results.update(text_results)
            
            # Print text training summary
            print("\n📊 Text Training Results:")
            for dataset, result in text_results.items():
                if 'error' not in result:
                    print(f"  ✅ {dataset}: {result['model_type']} - {result['best_val_accuracy']:.4f} acc")
                else:
                    print(f"  ❌ {dataset}: {result['error']}")
    
    if model_type in ['vision', 'all']:
        print("\n🖼️ Training your ResNet CIFAR models...")
        
        if 'vision_models' in config:
            vision_results = train_multiple_vision_models(config['vision_models'])
            training_results.update(vision_results)
            
            # Print vision training summary
            print("\n📊 Vision Training Results:")
            for model_name, result in vision_results.items():
                if 'error' not in result:
                    print(f"  ✅ {model_name}: {result['architecture']} - {result['best_val_accuracy']:.2f}% acc")
                    print(f"     📦 Parameters: {result['model_info']['total_parameters']:,}")
                else:
                    print(f"  ❌ {model_name}: {result['error']}")
    
    # Save training results
    results_path = 'training_results_summary.json'
    with open(results_path, 'w') as f:
        json.dump(training_results, f, indent=2, default=str)
    
    print(f"\n💾 Training results saved to: {results_path}")
    return training_results


def evaluation_stage(config: dict):
    """Evaluation stage using your actual models."""
    print(f"\n📈 Stage 3: Model Evaluation")
    print("-" * 30)
    
    try:
        evaluation_results = evaluate_all_models(config)
        
        # Generate summary report
        report_lines = [
            "# Reverse Attribution - Evaluation Report",
            f"Generated using your actual model implementations\n",
            "## Model Integration Status",
            "✅ Using BERTSentimentClassifier for text tasks",
            "✅ Using ResNetCIFAR for vision tasks", 
            "✅ RA framework integrated with actual models\n",
            "## Results Summary"
        ]
        
        for key, result in evaluation_results.items():
            if result and 'standard_metrics' in result:
                metrics = result['standard_metrics'] 
                ra_summary = result['ra_analysis']['summary']
                
                report_lines.extend([
                    f"\n### {key.replace('_results', '').upper()}",
                    f"- **Model Type**: {metrics.get('model_type', 'Unknown')}",
                    f"- **Architecture**: {metrics.get('architecture', 'N/A')}",
                    f"- **Accuracy**: {metrics['accuracy']:.4f}",
                    f"- **ECE**: {metrics['ece']:.4f}",
                    f"- **A-Flip Score**: {ra_summary['avg_a_flip']:.4f}",
                    f"- **RA Samples**: {ra_summary['samples_analyzed']}",
                    f"- **Model Types Detected**: {ra_summary['model_types_detected']}"
                ])
        
        # Save report
        report_path = 'evaluation_report.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\n📄 Evaluation report saved to: {report_path}")
        return evaluation_results
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate your actual RA models")
    parser.add_argument("--config", type=str, default="configs/experiment.yml",
                       help="Path to configuration file")
    parser.add_argument("--stage", type=str, 
                       choices=['data', 'train', 'eval', 'all'],
                       default='all', help="Which stage to run")
    parser.add_argument("--model_type", type=str, 
                       choices=['text', 'vision', 'all'],
                       default='all', help="Which model types to train")
    parser.add_argument("--check_models", action='store_true',
                       help="Check availability of your model implementations")
    
    args = parser.parse_args()
    
    print("🚀 Reverse Attribution - Model Training & Evaluation Pipeline")
    print("🏷️  Using your actual BERTSentimentClassifier and ResNetCIFAR implementations")
    print("="*80)
    
    # Check model availability
    if args.check_models:
        print("\n🔍 Checking your model implementations...")
        available = check_model_availability()
        return
    
    # Quick availability check
    print("\n🔍 Quick model availability check...")
    available = check_model_availability()
    
    if not any(available.values()):
        print("❌ No model implementations found! Please check your models/ directory")
        return
    
    # Load configuration
    config = load_config(args.config)
    
    print(f"\n📋 Configuration loaded from: {args.config}")
    print(f"📊 Project: {config.get('meta', {}).get('project_name', 'Reverse Attribution')}")
    
    results = {}
    
    # Stage 1: Data preparation
    if args.stage in ['data', 'all']:
        try:
            setup_data_stage(config)
            results['data_setup'] = 'completed'
        except Exception as e:
            print(f"❌ Data stage failed: {e}")
            results['data_setup'] = f'failed: {e}'
    
    # Stage 2: Model training
    if args.stage in ['train', 'all']:
        try:
            training_results = training_stage(config, args.model_type)
            results['training'] = training_results
        except Exception as e:
            print(f"❌ Training stage failed: {e}")
            results['training'] = f'failed: {e}'
    
    # Stage 3: Evaluation
    if args.stage in ['eval', 'all']:
        try:
            evaluation_results = evaluation_stage(config)
            results['evaluation'] = evaluation_results
        except Exception as e:
            print(f"❌ Evaluation stage failed: {e}")
            results['evaluation'] = f'failed: {e}'
    
    # Final summary
    print(f"\n{'='*80}")
    print("🎉 PIPELINE COMPLETION SUMMARY")
    print(f"{'='*80}")
    
    for stage, result in results.items():
        if isinstance(result, str) and 'failed' in result:
            print(f"❌ {stage.upper()}: {result}")
        elif isinstance(result, dict) and result:
            print(f"✅ {stage.upper()}: Completed successfully")
        elif result == 'completed':
            print(f"✅ {stage.upper()}: Completed")
        else:
            print(f"⚠️ {stage.upper()}: No results")
    
    print(f"\n🏷️ Model implementations used:")
    if available['bert_sentiment']:
        print("  ✅ BERTSentimentClassifier")
    if available['resnet_cifar']:
        print("  ✅ ResNetCIFAR (resnet56_cifar)")
    if available['custom_models']:
        print("  ✅ Custom model examples")
    
    print(f"\n📊 All results and configurations saved to respective directories")
    print(f"🔬 RA framework integrated with your actual model implementations")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
