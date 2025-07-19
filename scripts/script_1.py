"""
Training script for text classification models (BERT, RoBERTa).
Implements the training procedures for IMDB and Yelp datasets.
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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ra.model_factory import ModelFactory
from ra.dataset_utils import DatasetLoader


def train_text_model(dataset_name: str, config: dict):
    """Train text classification model."""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = ModelFactory.create_text_model(
        model_name=config['model_name'],
        num_classes=config['num_classes']
    ).to(device)
    
    # Setup data
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
    
    # Setup training
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    
    total_steps = len(train_dataloader) * config['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Training loop
    model.train()
    best_val_acc = 0.0
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        print("-" * 30)
        
        # Training phase
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(train_dataloader, desc="Training")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{total_correct/total_samples:.4f}'
            })
        
        train_acc = total_correct / total_samples
        avg_train_loss = total_loss / len(train_dataloader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                val_correct += (predictions == labels).sum().item()
                val_samples += labels.size(0)
        
        val_acc = val_correct / val_samples
        avg_val_loss = val_loss / len(val_dataloader)
        
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }
            torch.save(checkpoint, os.path.join(config['output_dir'], 'best_model.pt'))
            print(f"New best model saved! Val Acc: {val_acc:.4f}")
        
        model.train()
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    # Example usage
    imdb_config = {
        'model_name': 'bert-base-uncased',
        'num_classes': 2,
        'epochs': 3,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'max_length': 512,
        'output_dir': './checkpoints/bert_imdb'
    }
    
    train_text_model("imdb", imdb_config)
