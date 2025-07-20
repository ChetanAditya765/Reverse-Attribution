# Add this to ra/model_utils.py (create if it doesn't exist)

import os
import torch
from pathlib import Path
from typing import Dict, Optional, Union

def unified_model_check(model_name: str, check_type: str = "availability") -> Dict[str, any]:
    """
    Unified model checking that works consistently across all parts of the codebase.

    Args:
        model_name: Name of the model to check ('bert_sentiment', 'resnet_cifar', etc.)
        check_type: Type of check ('availability', 'checkpoint', 'both')

    Returns:
        Dict with model status information
    """
    results = {
        'model_name': model_name,
        'available': False,
        'checkpoint_exists': False,
        'can_instantiate': False,
        'path': None,
        'error': None
    }

    try:
        # Check 1: Can we import and instantiate the model?
        if model_name == 'bert_sentiment':
            from models.bert_sentiment import BERTSentimentClassifier
            model = BERTSentimentClassifier("bert-base-uncased", num_classes=2)
            results['can_instantiate'] = True
            del model  # Free memory

        elif model_name == 'resnet_cifar':
            from models.resnet_cifar import resnet56_cifar
            model = resnet56_cifar(num_classes=10)
            results['can_instantiate'] = True
            del model  # Free memory

        # Check 2: Do we have saved checkpoints?
        checkpoint_paths = [
            f"models/{model_name}/pytorch_model.bin",
            f"models/{model_name}/model.pth",
            f"checkpoints/{model_name}/best_model.pth",
            f"checkpoints/{model_name}/pytorch_model.bin"
        ]

        for path in checkpoint_paths:
            if os.path.exists(path):
                results['checkpoint_exists'] = True
                results['path'] = path
                break

        # Overall availability - model is available if it can be instantiated
        # Checkpoint is optional for training from scratch
        results['available'] = results['can_instantiate']

    except Exception as e:
        results['error'] = str(e)
        results['available'] = False

    return results


def check_all_models() -> Dict[str, Dict]:
    """Check all models in the repository."""
    models_to_check = ['bert_sentiment', 'resnet_cifar', 'custom_models']
    results = {}

    for model_name in models_to_check:
        results[model_name] = unified_model_check(model_name)

    return results


# Updated model availability check for script.py
def check_model_availability_fixed():
    """Fixed version that doesn't produce contradictory results."""
    results = check_all_models()

    print("🔍 Model Availability Check (Unified):")
    print("=" * 50)

    for model_name, status in results.items():
        if status['available']:
            checkpoint_status = "✅ Has checkpoints" if status['checkpoint_exists'] else "⚪ No checkpoints (can train from scratch)"
            print(f"✅ {model_name}: Available")
            print(f"   - Can instantiate: {status['can_instantiate']}")
            print(f"   - {checkpoint_status}")
            if status['path']:
                print(f"   - Checkpoint path: {status['path']}")
        else:
            print(f"❌ {model_name}: Not available")
            if status['error']:
                print(f"   - Error: {status['error']}")

    return results
