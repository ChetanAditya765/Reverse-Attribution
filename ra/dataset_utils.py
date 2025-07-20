"""
Dataset utilities for loading and preprocessing data.
Supports IMDB, Yelp Polarity, and CIFAR-10 datasets.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from datasets import load_dataset
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import os


class TextDataset(Dataset):
    """Custom dataset for text classification tasks."""
    
    def __init__(
        self, 
        texts: List[str], 
        labels: List[int], 
        tokenizer=None,
        max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        if self.tokenizer:
            encoded = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors="pt"
            )
            return {
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        
        return {'text': text, 'labels': torch.tensor(label, dtype=torch.long)}


class DatasetLoader:
    """Main class for loading different datasets."""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def load_imdb(self, split: str = "train") -> Tuple[List[str], List[int]]:
        """Load IMDB sentiment analysis dataset."""
        dataset = load_dataset("imdb", cache_dir=self.data_dir)
        
        texts = dataset[split]['text']
        labels = dataset[split]['label']
        
        return texts, labels
    
    def load_yelp_polarity(self, split: str = "train") -> Tuple[List[str], List[int]]:
        """Load Yelp polarity dataset."""
        dataset = load_dataset("yelp_polarity", cache_dir=self.data_dir)
        
        texts = dataset[split]['text']
        labels = dataset[split]['label']
        
        return texts, labels
    
    def load_cifar10(
        self, 
        split: str = "train",
        transform: Optional[transforms.Compose] = None
    ) -> datasets.CIFAR10:
        """Load CIFAR-10 dataset."""
        if transform is None:
            if split == "train":
                transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                       (0.2023, 0.1994, 0.2010))
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                       (0.2023, 0.1994, 0.2010))
                ])
        
        is_train = (split == "train")
        dataset = datasets.CIFAR10(
            root=os.path.join(self.data_dir, "cifar10"),
            train=is_train,
            download=True,
            transform=transform
        )
        
        return dataset
    
    def create_text_dataloader(
        self,
        dataset_name: str,
        split: str,
        tokenizer,
        batch_size: int = 32,
        max_length: int = 512,
        shuffle: bool = True
    ) -> DataLoader:
        """Create DataLoader for text datasets."""
        if dataset_name.lower() == "imdb":
            texts, labels = self.load_imdb(split)
        elif dataset_name.lower() == "yelp":
            texts, labels = self.load_yelp_polarity(split)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset = TextDataset(texts, labels, tokenizer, max_length)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4 if torch.cuda.is_available() else 0,
            pin_memory=True
        )
    
    def create_vision_dataloader(
        self,
        split: str,
        batch_size: int = 128,
        shuffle: bool = True,
        transform: Optional[transforms.Compose] = None
    ) -> DataLoader:
        """Create DataLoader for CIFAR-10."""
        dataset = self.load_cifar10(split, transform)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4 if torch.cuda.is_available() else 0,
            pin_memory=True
        )


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """Get information about dataset."""
    info = {
        "imdb": {
            "num_classes": 2,
            "class_names": ["negative", "positive"],
            "task": "sentiment_analysis",
            "domain": "text"
        },
        "yelp": {
            "num_classes": 2,
            "class_names": ["negative", "positive"],
            "task": "sentiment_analysis",
            "domain": "text"
        },
        "cifar10": {
            "num_classes": 10,
            "class_names": ["airplane", "automobile", "bird", "cat", "deer",
                          "dog", "frog", "horse", "ship", "truck"],
            "task": "image_classification",
            "domain": "vision"
        }
    }
    
    return info.get(dataset_name.lower(), {})
