"""
Training script for vision models (ResNet-56 on CIFAR-10).
Implements the training procedure described in the paper.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ra.model_factory import ModelFactory
from ra.dataset_utils import DatasetLoader


def train_vision_model(config: dict):
    """Train vision model (ResNet-56) on CIFAR-10."""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = ModelFactory.create_vision_model(
        num_classes=config['num_classes'],
        architecture=config['architecture']
    ).to(device)
    
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
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['learning_rate'],
        momentum=0.9,
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    milestones = [config['epochs'] // 2, 3 * config['epochs'] // 4]
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=milestones, 
        gamma=0.1
    )
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        print("-" * 30)
        
        # Training phase
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(train_dataloader, desc="Training")
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += target.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{total_correct/total_samples:.4f}'
            })
        
        train_acc = total_correct / total_samples
        avg_train_loss = total_loss / len(train_dataloader)
        
        # Update learning rate
        scheduler.step()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_samples = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_dataloader, desc="Validation"):
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()
                val_samples += target.size(0)
        
        val_acc = val_correct / val_samples
        avg_val_loss = val_loss / len(val_dataloader)
        
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Current LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'config': config
            }
            torch.save(checkpoint, os.path.join(config['output_dir'], 'best_model.pt'))
            print(f"New best model saved! Val Acc: {val_acc:.4f}")
        
        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            checkpoint_path = os.path.join(config['output_dir'], f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, checkpoint_path)
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    # Example usage
    cifar10_config = {
        'architecture': 'resnet56',
        'num_classes': 10,
        'epochs': 200,
        'batch_size': 128,
        'learning_rate': 0.1,
        'weight_decay': 1e-4,
        'output_dir': './checkpoints/resnet56_cifar10'
    }
    
    train_vision_model(cifar10_config)
