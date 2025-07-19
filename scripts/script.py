"""
Main training script that orchestrates the training of all models
used in the Reverse Attribution paper experiments.
"""

import argparse
import yaml
import os
import torch
from pathlib import Path

# Add parent directory to path to import from ra module
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ra.model_factory import ModelFactory
from ra.dataset_utils import DatasetLoader
from scripts.script_1 import train_text_model
from scripts.script_2 import train_vision_model  
from scripts.script_3 import evaluate_all_models


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train all models for RA experiments")
    parser.add_argument("--config", type=str, default="configs/experiment.yml",
                       help="Path to configuration file")
    parser.add_argument("--stage", type=str, choices=['data', 'train', 'eval', 'all'],
                       default='all', help="Which stage to run")
    parser.add_argument("--model_type", type=str, choices=['text', 'vision', 'all'],
                       default='all', help="Which models to train")
    
    args = parser.parse_args()
    
    # Create default config if it doesn't exist
    config_dir = os.path.dirname(args.config)
    os.makedirs(config_dir, exist_ok=True)
    
    if not os.path.exists(args.config):
        default_config = {
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
                    'output_dir': './checkpoints/bert_imdb'
                },
                'yelp': {
                    'model_name': 'roberta-large',
                    'num_classes': 2, 
                    'epochs': 3,
                    'batch_size': 8,
                    'learning_rate': 1e-5,
                    'max_length': 512,
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
                    'weight_decay': 1e-4,
                    'output_dir': './checkpoints/resnet56_cifar10'
                }
            }
        }
        
        with open(args.config, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        print(f"Created default config at {args.config}")
    
    config = load_config(args.config)
    
    print("🚀 Starting Reverse Attribution model training pipeline")
    print("=" * 60)
    
    # Stage 1: Data preparation
    if args.stage in ['data', 'all']:
        print("\n📊 Stage 1: Data Preparation")
        from ra.download_datasets import download_all_datasets
        from ra.download_models import download_pretrained_models
        
        if config['data']['download']:
            download_all_datasets(config['data']['data_dir'])
            download_pretrained_models()
    
    # Stage 2: Model training
    if args.stage in ['train', 'all']:
        print("\n🏋️ Stage 2: Model Training")
        
        if args.model_type in ['text', 'all']:
            print("\n📚 Training text models...")
            
            # Train BERT on IMDB
            if 'imdb' in config['text_models']:
                print("  Training BERT on IMDB...")
                train_text_model(
                    dataset_name="imdb",
                    config=config['text_models']['imdb']
                )
            
            # Train RoBERTa on Yelp
            if 'yelp' in config['text_models']:
                print("  Training RoBERTa on Yelp...")
                train_text_model(
                    dataset_name="yelp", 
                    config=config['text_models']['yelp']
                )
        
        if args.model_type in ['vision', 'all']:
            print("\n🖼️ Training vision models...")
            
            # Train ResNet-56 on CIFAR-10
            if 'cifar10' in config['vision_models']:
                print("  Training ResNet-56 on CIFAR-10...")
                train_vision_model(config['vision_models']['cifar10'])
    
    # Stage 3: Evaluation
    if args.stage in ['eval', 'all']:
        print("\n📈 Stage 3: Model Evaluation")
        evaluate_all_models(config)
    
    print("\n✅ Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
