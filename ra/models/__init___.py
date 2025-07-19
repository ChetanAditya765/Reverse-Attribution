"""
models/__init__.py

Model implementations for the Reverse Attribution framework.

This package exposes:
- BERT-based sentiment analysis models
- Custom model integration examples
- ResNet architectures for CIFAR datasets
"""

# BERT sentiment models
from .bert_sentiment import (
    BERTSentimentClassifier,
    BERTSentimentTrainer,
    create_bert_sentiment_model,
)

# Custom model examples
from .custom_model_example import (
    CustomTextClassifier,
    CustomVisionClassifier,
    CustomModelWrapper,
    demonstrate_custom_text_model,
    demonstrate_custom_vision_model,
    run_complete_example,
)

# ResNet CIFAR models
from .resnet_cifar import (
    ResNetCIFAR,
    BasicBlock,
    Bottleneck,
    ResNetCIFARTrainer,
    resnet20_cifar,
    resnet32_cifar,
    resnet44_cifar,
    resnet56_cifar,
    resnet110_cifar,
    wide_resnet28_10_cifar,
    get_model_info,
)

# Convenience factory functions
def get_bert_model(model_name: str = "bert-base-uncased", num_classes: int = 2, **kwargs):
    """
    Create a BERT sentiment model.
    """
    return create_bert_sentiment_model(model_name, num_classes, **kwargs)

def get_resnet56_model(num_classes: int = 10, **kwargs):
    """
    Create a ResNet-56 model for CIFAR.
    """
    return resnet56_cifar(num_classes, **kwargs)

__all__ = [
    # BERT models
    "BERTSentimentClassifier",
    "BERTSentimentTrainer",
    "create_bert_sentiment_model",
    "get_bert_model",
    # Custom models
    "CustomTextClassifier",
    "CustomVisionClassifier",
    "CustomModelWrapper",
    "demonstrate_custom_text_model",
    "demonstrate_custom_vision_model",
    "run_complete_example",
    # ResNet CIFAR
    "ResNetCIFAR",
    "BasicBlock",
    "Bottleneck",
    "ResNetCIFARTrainer",
    "resnet20_cifar",
    "resnet32_cifar",
    "resnet44_cifar",
    "resnet56_cifar",
    "resnet110_cifar",
    "wide_resnet28_10_cifar",
    "get_model_info",
    "get_resnet56_model",
]
