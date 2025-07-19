"""
Training script for your actual ResNet CIFAR models.
Now properly uses your ResNetCIFAR implementation and trainer.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import your actual ResNet implementations
from ra.models.resnet_cifar import (
    resnet20_cifar, resnet32_cifar, resnet56_cifar, resnet110_cifar,
    ResNetCIFARTrainer, get_model_info
)
from ra.dataset_utils import DatasetLoader


def train_vision_model(config: dict):
    """Train your actual ResNet CIFAR model."""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    architecture = config.get('architecture', 'resnet56')
    
    print(f"🚀 Training {architecture} on CIFAR-10")
    print(f"Using device: {device}")
    
    # Create your actual ResNet model
    architecture_map = {
        'resnet20': resnet20_cifar,
        'resnet32': resnet32_cifar,
        'resnet56': resnet56_cifar,
        'resnet110': resnet110_cifar
    }
    
    if architecture not in architecture_map:
        raise ValueError(f"Unsupported architecture: {architecture}. Choose from {list(architecture_map.keys())}")
    
    model = architecture_map[architecture](
        num_classes=config['num_classes'],
        zero_init_residual=config.get('zero_init_residual', False)
    ).to(device)
    
    print(f"✅ Created {model.__class__.__name__} ({architecture})")
    
    # Get model info using your utility
    model_info = get_model_info(model)
    print(f"📊 Total parameters: {model_info['total_parameters']:,}")
    print(f"🔧 Model size: {model_info['model_size_mb']:.1f} MB")
    print(f"🏗️ Total layers: {model_info['total_layers']}")
    
    # Setup data
    loader = DatasetLoader(config.get('data_dir', './data'))
    
    train_dataloader = loader.create_vision_dataloader(
        split="train",
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    val_dataloader = loader.create_vision_dataloader(
        split="test", 
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    print(f"📚 Training batches: {len(train_dataloader)}")
    print(f"🔍 Validation batches: {len(val_dataloader)}")
    
    # Use your ResNet trainer
    trainer = ResNetCIFARTrainer(
        model=model,
        learning_rate=config['learning_rate'],
        momentum=config.get('momentum', 0.9),
        weight_decay=config['weight_decay'],
        milestones=config.get('milestones', [100, 150])
    )
    
    print(f"🎯 Initial LR: {trainer.learning_rate}")
    print(f"📅 LR milestones: {config.get('milestones', [100, 150])}")
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Training loop using your trainer
    best_val_acc = 0.0
    training_history = []
    
    for epoch in range(config['epochs']):
        if epoch % 10 == 0 or epoch < 5:  # Print every 10 epochs, plus first 5
            print(f"\n{'='*50}")
            print(f"EPOCH {epoch + 1}/{config['epochs']}")
            print(f"{'='*50}")
        
        # Training phase using your trainer
        train_metrics = trainer.train_epoch(train_dataloader, device)
        
        # Validation phase using your trainer  
        val_metrics = trainer.evaluate(val_dataloader, device)
        
        # Update learning rate scheduler
        trainer.scheduler.step()
        
        # Log metrics
        current_lr = trainer.scheduler.get_last_lr()[0]
        epoch_results = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_accuracy': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'learning_rate': current_lr
        }
        training_history.append(epoch_results)
        
        if epoch % 10 == 0 or epoch < 5:  # Print details every 10 epochs
            print(f"📊 Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"📈 Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
            print(f"🎯 Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict(),
                'val_acc': val_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'config': config,
                'model_info': model_info,
                'architecture': architecture,
                'training_history': training_history
            }
            
            best_model_path = os.path.join(config['output_dir'], 'best_model.pt')
            torch.save(checkpoint, best_model_path)
            
            if epoch % 10 == 0 or val_metrics['accuracy'] > best_val_acc:
                print(f"💾 New best model saved! Val Acc: {val_metrics['accuracy']:.2f}%")
        
        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            checkpoint_path = os.path.join(config['output_dir'], f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict(),
                'val_acc': val_metrics['accuracy'],
                'config': config,
                'model_info': model_info,
                'training_history': training_history
            }, checkpoint_path)
            print(f"📁 Checkpoint saved: checkpoint_epoch_{epoch+1}.pt")
        
        # Early stopping check (optional)
        if config.get('early_stopping', False) and epoch > 50:
            recent_accs = [h['val_accuracy'] for h in training_history[-10:]]
            if max(recent_accs) - min(recent_accs) < 0.1:  # Less than 0.1% improvement in 10 epochs
                print(f"🛑 Early stopping triggered at epoch {epoch+1}")
                break
    
    # Save final training history
    history_path = os.path.join(config['output_dir'], 'training_history.json')
    import json
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"🎉 Training completed!")
    print(f"🏗️ Architecture: {architecture}")
    print(f"📊 Best validation accuracy: {best_val_acc:.2f}%")
    print(f"💾 Best model saved to: {best_model_path}")
    print(f"📈 Training history saved to: {history_path}")
    print(f"🏷️ Model class: {model.__class__.__name__}")
    print(f"📦 Total parameters: {model_info['total_parameters']:,}")
    print(f"{'='*60}")
    
    return {
        'best_val_accuracy': best_val_acc,
        'model_path': best_model_path,
        'architecture': architecture,
        'model_type': model.__class__.__name__,
        'model_info': model_info,
        'training_history': training_history
    }


def train_multiple_vision_models(models_config: dict):
    """Train multiple vision models with your implementations."""
    
    results = {}
    
    for model_name, config in models_config.items():
        print(f"\n🚀 Starting training for {model_name}")
        try:
            result = train_vision_model(config)
            results[model_name] = result
            print(f"✅ Successfully trained {model_name} model")
        except Exception as e:
            print(f"❌ Failed to train {model_name} model: {e}")
            results[model_name] = {'error': str(e)}
    
    return results


if __name__ == "__main__":
    # Example usage with your ResNet models
    
    # ResNet-56 configuration (main model from paper)
    resnet56_config = {
        'architecture': 'resnet56',
        'num_classes': 10,
        'epochs': 200,
        'batch_size': 128,
        'learning_rate': 0.1,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'milestones': [100, 150],  # LR decay at these epochs
        'zero_init_residual': False,
        'output_dir': './checkpoints/resnet56_cifar10',
        'data_dir': './data'
    }
    
    # ResNet-20 configuration (lighter model)
    resnet20_config = {
        'architecture': 'resnet20',
        'num_classes': 10,
        'epochs': 200,
        'batch_size': 128,
        'learning_rate': 0.1,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'milestones': [100, 150],
        'output_dir': './checkpoints/resnet20_cifar10',
        'data_dir': './data'
    }
    
    # Train single model (ResNet-56 as used in paper)
    print("🎯 Training ResNet-56 with your implementation...")
    resnet56_result = train_vision_model(resnet56_config)
    
    # Or train multiple architectures
    models_config = {
        'resnet56': resnet56_config,
        'resnet20': resnet20_config
    }
    
    print("\n🎯 Training all vision models with your implementations...")
    all_results = train_multiple_vision_models(models_config)
    
    print("\n📊 TRAINING SUMMARY")
    print("="*60)
    for model_name, result in all_results.items():
        if 'error' not in result:
            print(f"{model_name.upper()}:")
            print(f"  ✅ Architecture: {result['architecture']}")
            print(f"  🏷️ Model Type: {result['model_type']}")
            print(f"  📦 Parameters: {result['model_info']['total_parameters']:,}")
            print(f"  📈 Best Accuracy: {result['best_val_accuracy']:.2f}%")
            print(f"  💾 Saved to: {result['model_path']}")
        else:
            print(f"{model_name.upper()}: ❌ {result['error']}")
