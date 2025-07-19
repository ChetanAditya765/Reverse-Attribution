"""
Model Factory for creating and loading different model architectures.
Supports BERT, RoBERTa for text and ResNet for vision tasks.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from torchvision import models
import os
from typing import Tuple, Optional


class TextClassifier(nn.Module):
    """Wrapper for HuggingFace text classification models."""
    
    def __init__(self, model_name: str, num_classes: int, max_length: int = 512):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_classes
        )
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
    
    def encode_text(self, texts):
        """Tokenize and encode text inputs."""
        if isinstance(texts, str):
            texts = [texts]
            
        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return encoded


class VisionClassifier(nn.Module):
    """ResNet-based classifier for vision tasks."""
    
    def __init__(self, num_classes: int = 10, architecture: str = "resnet56"):
        super().__init__()
        
        if architecture == "resnet56":
            self.model = self._make_resnet56(num_classes)
        else:
            # Use pretrained ResNet from torchvision
            self.model = models.resnet50(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)
    
    def _make_resnet56(self, num_classes: int):
        """Create ResNet-56 architecture for CIFAR-10."""
        
        class BasicBlock(nn.Module):
            expansion = 1
            
            def __init__(self, in_planes, planes, stride=1):
                super().__init__()
                self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, 
                                     stride=stride, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(planes)
                self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                     stride=1, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(planes)
                
                self.shortcut = nn.Sequential()
                if stride != 1 or in_planes != planes:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_planes, planes, kernel_size=1,
                                stride=stride, bias=False),
                        nn.BatchNorm2d(planes)
                    )
            
            def forward(self, x):
                out = torch.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += self.shortcut(x)
                return torch.relu(out)
        
        class ResNet(nn.Module):
            def __init__(self, block, num_blocks, num_classes):
                super().__init__()
                self.in_planes = 16
                
                self.conv1 = nn.Conv2d(3, 16, kernel_size=3, 
                                     stride=1, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(16)
                self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
                self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
                self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
                self.linear = nn.Linear(64, num_classes)
            
            def _make_layer(self, block, planes, num_blocks, stride):
                strides = [stride] + [1] * (num_blocks - 1)
                layers = []
                for stride in strides:
                    layers.append(block(self.in_planes, planes, stride))
                    self.in_planes = planes * block.expansion
                return nn.Sequential(*layers)
            
            def forward(self, x):
                out = torch.relu(self.bn1(self.conv1(x)))
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = torch.avg_pool2d(out, 8)
                out = out.view(out.size(0), -1)
                out = self.linear(out)
                return out
        
        return ResNet(BasicBlock, [9, 9, 9], num_classes)


class ModelFactory:
    """Factory class for creating different model types."""
    
    @staticmethod
    def create_text_model(
        model_name: str = "bert-base-uncased",
        num_classes: int = 2,
        checkpoint_path: Optional[str] = None
    ) -> TextClassifier:
        """Create text classification model."""
        model = TextClassifier(model_name, num_classes)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
            
        return model
    
    @staticmethod
    def create_vision_model(
        num_classes: int = 10,
        architecture: str = "resnet56",
        checkpoint_path: Optional[str] = None
    ) -> VisionClassifier:
        """Create vision classification model."""
        model = VisionClassifier(num_classes, architecture)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
            
        return model
    
    @staticmethod
    def load_model(model_type: str, **kwargs) -> nn.Module:
        """Load model based on type specification."""
        if model_type.lower() in ['bert', 'roberta', 'text']:
            return ModelFactory.create_text_model(**kwargs)
        elif model_type.lower() in ['resnet', 'vision', 'cnn']:
            return ModelFactory.create_vision_model(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
