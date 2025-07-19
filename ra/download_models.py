"""
Script to download pretrained models used in the experiments.
Downloads BERT, RoBERTa checkpoints and sets up model directories.
"""

import os
import argparse
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from model_factory import ModelFactory


def download_pretrained_models(model_dir: str = "./models"):
    """Download all pretrained models used in the paper."""
    print("🔄 Starting model downloads...")
    
    os.makedirs(model_dir, exist_ok=True)
    
    models_to_download = [
        {
            "name": "bert-base-uncased",
            "type": "text",
            "description": "BERT base model for IMDB"
        },
        {
            "name": "roberta-large",
            "type": "text", 
            "description": "RoBERTa large for Yelp"
        }
    ]
    
    for model_info in models_to_download:
        try:
            print(f"\n📥 Downloading {model_info['description']}...")
            
            # Download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_info["name"],
                cache_dir=model_dir
            )
            
            # Download model
            model = AutoModelForSequenceClassification.from_pretrained(
                model_info["name"],
                cache_dir=model_dir
            )
            
            print(f"  ✅ {model_info['name']} downloaded successfully")
            
        except Exception as e:
            print(f"  ❌ Error downloading {model_info['name']}: {e}")
    
    print(f"\n🎉 Model downloads completed! Models cached in: {model_dir}")


def create_model_directories(base_dir: str = "./checkpoints"):
    """Create directory structure for saving trained models."""
    directories = [
        "bert_imdb",
        "roberta_yelp", 
        "resnet56_cifar10"
    ]
    
    for dir_name in directories:
        dir_path = os.path.join(base_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pretrained models")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./models", 
        help="Directory to cache models"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Directory for saving fine-tuned models"
    )
    
    args = parser.parse_args()
    
    download_pretrained_models(args.model_dir)
    create_model_directories(args.checkpoint_dir)
