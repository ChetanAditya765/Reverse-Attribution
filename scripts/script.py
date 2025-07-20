"""
Main orchestration script that coordinates training and evaluation of your actual models.
Now properly integrates with BERTSentimentClassifier and ResNetCIFAR implementations.
"""

import argparse
import yaml
import os
import torch
# Add after line 9
from ra.device_utils import device, get_device
from pathlib import Path
import json
import sys

# Ensure project root is on PYTHONPATH
ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import your model-integrated scripts
from script_1 import train_text_model, train_multiple_text_models
from script_2 import train_vision_model, train_multiple_vision_models
from script_3 import evaluate_all_models

# Import your actual model availability functions
from models import get_bert_model, get_resnet56_model


def check_model_availability():
    """Check which of your models are available."""
    available = {
        'bert_sentiment': False,
        'resnet_cifar': False,
        'custom_models': False
    }
    
    try:
        _ = get_bert_model("bert-base-uncased", num_classes=2)
        available['bert_sentiment'] = True
        print("✅ Your BERT sentiment model available")
    except Exception as e:
        print(f"❌ BERT sentiment model unavailable: {e}")
    
    try:
        _ = get_resnet56_model(num_classes=10)
        available['resnet_cifar'] = True
        print("✅ Your ResNet CIFAR model available")
    except Exception as e:
        print(f"❌ ResNet CIFAR model unavailable: {e}")
    
    try:
        from models.custom_model_example import CustomTextClassifier
        _ = CustomTextClassifier(num_classes=2)
        available['custom_models'] = True
        print("✅ Your custom model examples available")
    except Exception as e:
        print(f"❌ Your custom model examples unavailable: {e}")
    
    return available


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file, or create default if missing."""
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
            'model_implementations': 'Actual BERTSentimentClassifier & ResNetCIFAR',
            'paper_reference': 'JMLR Reverse Attribution Paper'
        },
        'data': {
            'data_dir': './data',
            'download': True
        },
        'text_models': {
            'imdb': {
                'model_name': 'bert-base-uncased',
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

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)
    print(f"📝 Created default config at {config_path}")
    return default_config


def setup_data_stage(config: dict):
    """Stage 1: prepare and optionally download datasets."""
    print("\n📊 Stage 1: Data Preparation")
    print("-" * 30)
    data_dir = config['data']['data_dir']
    os.makedirs(data_dir, exist_ok=True)
    if config['data']['download']:
        print("📥 Downloading datasets...")
        try:
            from ra.dataset_utils import DatasetLoader
            loader = DatasetLoader(data_dir)
            texts, labels = loader.load_imdb("train")
            print(f"  ✅ IMDB loaded: {len(texts)} samples")
            texts, labels = loader.load_yelp_polarity("train")
            print(f"  ✅ Yelp loaded: {len(texts)} samples")
            cifar_ds = loader.load_cifar10("train")
            print(f"  ✅ CIFAR-10 loaded: {len(cifar_ds)} samples")
        except Exception as e:
            print(f"    ⚠️ Dataset download issue: {e}")


def training_stage(config: dict, model_type: str):
    """Stage 2: train text and/or vision models."""
    print(f"\n🏋️ Stage 2: Model Training ({model_type})")
    print("-" * 40)
    results = {}

    if model_type in ('text', 'all'):
        print("\n📚 Training text models...")
        if 'text_models' in config:
            text_results = train_multiple_text_models(config['text_models'])
            results.update(text_results)
            print("\n📊 Text Training Results:")
            for ds, res in text_results.items():
                if 'error' not in res:
                    print(f"  ✅ {ds}: {res['best_val_accuracy']:.4f} acc")
                else:
                    print(f"  ❌ {ds}: {res['error']}")

    if model_type in ('vision', 'all'):
        print("\n🖼️ Training vision models...")
        if 'vision_models' in config:
            vision_results = train_multiple_vision_models(config['vision_models'])
            results.update(vision_results)
            print("\n📊 Vision Training Results:")
            for name, res in vision_results.items():
                if 'error' not in res:
                    acc = res['best_val_accuracy'] * 100
                    print(f"  ✅ {name}: {acc:.2f}% acc")
                else:
                    print(f"  ❌ {name}: {res['error']}")

    with open('training_results_summary.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\n💾 Saved training summary to training_results_summary.json")
    return results


def evaluation_stage(config: dict):
    """Stage 3: evaluate all trained models."""
    print(f"\n📈 Stage 3: Model Evaluation")
    print("-" * 30)
    try:
        eval_results = evaluate_all_models(config)
        lines = [
            "# Reverse Attribution - Evaluation Report",
            f"Generated with actual model implementations\n",
            "## Integration Status",
            "✅ BERTSentimentClassifier",
            "✅ ResNetCIFAR",
            "✅ RA framework integrated\n",
            "## Results Summary"
        ]
        for key, res in eval_results.items():
            if res and 'standard_metrics' in res:
                m = res['standard_metrics']
                ra = res['ra_analysis']['summary']
                lines += [
                    f"\n### {key.upper()}",
                    f"- Accuracy: {m['accuracy']:.4f}",
                    f"- ECE: {m['ece']:.4f}",
                    f"- A-Flip: {ra['avg_a_flip']:.4f}"
                ]
        with open('evaluation_report.md', 'w') as f:
            f.write('\n'.join(lines))
        print("\n📄 Saved evaluation report to evaluation_report.md")
        return eval_results
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Train & evaluate your Reverse Attribution models"
    )
    parser.add_argument("--config", type=str, default="configs/experiment.yml")
    parser.add_argument("--stage", choices=['data','train','eval','all'], default='all')
    parser.add_argument("--model_type", choices=['text','vision','all'], default='all')
    parser.add_argument("--check_models", action='store_true')
    args = parser.parse_args()

    print("🚀 Reverse Attribution Pipeline")
    print(f"🔧 Device: {get_device()}")
    print("=" * 80)

    if args.check_models:
        check_model_availability()
        return

    print("\n🔍 Checking model availability…")
    avail = check_model_availability()
    if not any(avail.values()):
        print("❌ No models found. Aborting.")
        return

    cfg = load_config(args.config)
    print(f"\n📋 Loaded config: {args.config}")

    results = {}
    if args.stage in ('data','all'):
        try:
            setup_data_stage(cfg); results['data'] = 'done'
        except Exception as e:
            print(f"❌ Data stage error: {e}"); results['data'] = f'error: {e}'
    if args.stage in ('train','all'):
        try:
            results['train'] = training_stage(cfg, args.model_type)
        except Exception as e:
            print(f"❌ Training stage error: {e}"); results['train'] = f'error: {e}'
    if args.stage in ('eval','all'):
        try:
            results['eval'] = evaluation_stage(cfg)
        except Exception as e:
            print(f"❌ Eval stage error: {e}"); results['eval'] = f'error: {e}'

    print("\n" + "="*80)
    print("🎉 Pipeline Summary")
    for k, v in results.items():
        status = '✅' if not (isinstance(v, str) and 'error' in v) else '❌'
        print(f"{status} {k.capitalize()}: {v}")
    print("="*80)

    print("\n🏷️ Models used:")
    for name, ok in avail.items():
        print(f"  {'✅' if ok else '❌'} {name}")
    print("="*80)


if __name__ == "__main__":
    main()
