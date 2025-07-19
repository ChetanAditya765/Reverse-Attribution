"""
Training script for your actual BERT sentiment models.
Now properly uses BERTSentimentClassifier and BERTSentimentTrainer.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import os
import numpy as np
from tqdm import tqdm
import logging
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import your actual BERT model implementation
from ra.models.bert_sentiment import BERTSentimentClassifier, BERTSentimentTrainer
from ra.dataset_utils import DatasetLoader


def train_text_model(dataset_name: str, config: dict):
    """Train your actual BERT sentiment model."""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training {config['model_name']} on {dataset_name}")
    print(f"Using device: {device}")
    
    # Create your actual BERT sentiment model
    model = BERTSentimentClassifier(
        model_name=config['model_name'],
        num_classes=config['num_classes'],
        dropout_rate=config.get('dropout_rate', 0.1),
        freeze_bert=config.get('freeze_bert', False),
        use_pooler=config.get('use_pooler', True)
    ).to(device)
    
    print(f"✅ Created {model.__class__.__name__}")
    model_info = model.get_model_info()
    print(f"📊 Total parameters: {model_info['total_parameters']:,}")
    print(f"🔧 Trainable parameters: {model_info['trainable_parameters']:,}")
    
    # Setup data using your dataset utilities
    loader = DatasetLoader(config.get('data_dir', './data'))
    
    train_dataloader = loader.create_text_dataloader(
        dataset_name=dataset_name,
        split="train",
        tokenizer=model.tokenizer,
        batch_size=config['batch_size'],
        max_length=config['max_length']
    )
    
    val_dataloader = loader.create_text_dataloader(
        dataset_name=dataset_name, 
        split="test",
        tokenizer=model.tokenizer,
        batch_size=config['batch_size'],
        max_length=config['max_length'],
        shuffle=False
    )
    
    print(f"📚 Training batches: {len(train_dataloader)}")
    print(f"🔍 Validation batches: {len(val_dataloader)}")
    
    # Use your BERTSentimentTrainer
    trainer = BERTSentimentTrainer(
        model=model,
        learning_rate=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.01),
        warmup_steps=config.get('warmup_steps', 0)
    )
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Training loop using your trainer
    best_val_acc = 0.0
    training_history = []
    
    for epoch in range(config['epochs']):
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch + 1}/{config['epochs']}")
        print(f"{'='*50}")
        
        # Training phase using your trainer
        print("🏋️ Training...")
        train_metrics = trainer.train_epoch(train_dataloader, device)
        
        # Validation phase using your trainer
        print("🔍 Validating...")
        val_metrics = trainer.evaluate(val_dataloader, device)
        
        # Log metrics
        epoch_results = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_accuracy': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'learning_rate': trainer.optimizer.param_groups[0]['lr']
        }
        training_history.append(epoch_results)
        
        print(f"📊 Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"📈 Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"🎯 Learning Rate: {trainer.optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_acc': val_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'config': config,
                'model_info': model_info,
                'training_history': training_history
            }
            
            best_model_path = os.path.join(config['output_dir'], 'best_model.pt')
            torch.save(checkpoint, best_model_path)
            print(f"💾 New best model saved! Val Acc: {val_metrics['accuracy']:.4f}")
        
        # Save epoch checkpoint
        if (epoch + 1) % 5 == 0:  # Save every 5 epochs
            epoch_checkpoint_path = os.path.join(config['output_dir'], f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': None,
                'config': config,
                'training_history': training_history
            }, epoch_checkpoint_path)
    
    # Save final training history
    history_path = os.path.join(config['output_dir'], 'training_history.json')
    import json
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"🎉 Training completed!")
    print(f"📊 Best validation accuracy: {best_val_acc:.4f}")
    print(f"💾 Best model saved to: {best_model_path}")
    print(f"📈 Training history saved to: {history_path}")
    print(f"🏷️ Model type: {model.__class__.__name__}")
    print(f"{'='*50}")
    
    return {
        'best_val_accuracy': best_val_acc,
        'model_path': best_model_path,
        'model_type': model.__class__.__name__,
        'training_history': training_history
    }


def train_multiple_text_models(models_config: dict):
    """Train multiple text models with your implementations."""
    
    results = {}
    
    for dataset_name, config in models_config.items():
        print(f"\n🚀 Starting training for {dataset_name}")
        try:
            result = train_text_model(dataset_name, config)
            results[dataset_name] = result
            print(f"✅ Successfully trained {dataset_name} model")
        except Exception as e:
            print(f"❌ Failed to train {dataset_name} model: {e}")
            results[dataset_name] = {'error': str(e)}
    
    return results


if __name__ == "__main__":
    # Example usage with your BERT models
    
    # IMDB configuration for your BERT model
    imdb_config = {
        'model_name': 'bert-base-uncased',
        'num_classes': 2,
        'epochs': 3,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'max_length': 512,
        'dropout_rate': 0.1,
        'weight_decay': 0.01,
        'output_dir': './checkpoints/bert_imdb',
        'data_dir': './data'
    }
    
    # Yelp configuration for your BERT model  
    yelp_config = {
        'model_name': 'roberta-base',  # or roberta-large if you have resources
        'num_classes': 2,
        'epochs': 3,
        'batch_size': 8,
        'learning_rate': 1e-5,
        'max_length': 512,
        'dropout_rate': 0.1,
        'weight_decay': 0.01,
        'output_dir': './checkpoints/roberta_yelp',
        'data_dir': './data'
    }
    
    # Train individual model
    print("🎯 Training BERT on IMDB with your implementation...")
    imdb_result = train_text_model("imdb", imdb_config)
    
    # Or train multiple models
    models_config = {
        'imdb': imdb_config,
        'yelp': yelp_config
    }
    
    print("\n🎯 Training all text models with your implementations...")
    all_results = train_multiple_text_models(models_config)
    
    print("\n📊 TRAINING SUMMARY")
    print("="*50)
    for dataset, result in all_results.items():
        if 'error' not in result:
            print(f"{dataset.upper()}:")
            print(f"  ✅ Model Type: {result['model_type']}")
            print(f"  📈 Best Accuracy: {result['best_val_accuracy']:.4f}")
            print(f"  💾 Saved to: {result['model_path']}")
        else:
            print(f"{dataset.upper()}: ❌ {result['error']}")
