# models/__init__.py

"""
Top-level models package for Reverse Attribution.

Re-exports implementations from ra.models so that scripts
can import via `models.*` without changing existing code.
"""

# BERT-based sentiment models
from models.bert_sentiment import (
    BERTSentimentClassifier,
    BERTSentimentTrainer,
    create_bert_sentiment_model,
)

# ResNet CIFAR implementations
from models.resnet_cifar import (
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

# Custom model examples
from models.custom_model_example import (
    CustomTextClassifier,
    CustomVisionClassifier,
    CustomModelWrapper,
    demonstrate_custom_text_model,
    demonstrate_custom_vision_model,
    run_complete_example,
)

# Convenience factory functions
def get_bert_model(model_name: str = "bert-base-uncased", num_classes: int = 2, **kwargs):
    """
    Create and return a BERTSentimentClassifier.
    """
    return create_bert_sentiment_model(model_name, num_classes, **kwargs)

def get_resnet56_model(num_classes: int = 10, **kwargs):
    """
    Create and return a ResNet-56 CIFAR model.
    """
    return resnet56_cifar(num_classes, **kwargs)

__all__ = [
    # BERT sentiment
    "BERTSentimentClassifier",
    "BERTSentimentTrainer",
    "create_bert_sentiment_model",
    "get_bert_model",
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
    # Custom examples
    "CustomTextClassifier",
    "CustomVisionClassifier",
    "CustomModelWrapper",
    "demonstrate_custom_text_model",
    "demonstrate_custom_vision_model",
    "run_complete_example",
]
