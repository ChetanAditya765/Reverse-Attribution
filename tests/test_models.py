"""
tests/test_models.py

Unit-tests for model creation and forward passes using your actual implementations.
"""

import torch
import pytest

# Import your actual model classes
from ra.models.bert_sentiment import BERTSentimentClassifier, create_bert_sentiment_model
from ra.models.resnet_cifar import (
    resnet20_cifar, resnet32_cifar, resnet44_cifar,
    resnet56_cifar, resnet110_cifar, get_model_info
)


@pytest.mark.parametrize("model_name", ["bert-base-uncased", "prajjwal1/bert-tiny"])
def test_bert_sentiment_forward(model_name):
    # Create model via factory and direct class
    model1 = create_bert_sentiment_model(model_name, num_classes=2)
    model2 = BERTSentimentClassifier(model_name, num_classes=2)
    
    # Dummy input
    input_ids = torch.randint(0, model1.tokenizer.vocab_size, (1, 16))
    attention_mask = torch.ones_like(input_ids)
    
    for model in (model1, model2):
        logits = model(input_ids, attention_mask)
        assert logits.shape == (1, 2)
        assert torch.isfinite(logits).all()


@pytest.mark.parametrize("arch, func", [
    ("resnet20", resnet20_cifar),
    ("resnet32", resnet32_cifar),
    ("resnet44", resnet44_cifar),
    ("resnet56", resnet56_cifar),
    ("resnet110", resnet110_cifar),
])
def test_resnet_cifar_forward_and_info(arch, func):
    model = func(num_classes=10)
    dummy = torch.randn(2, 3, 32, 32)
    logits = model(dummy)
    assert logits.shape == (2, 10)
    assert torch.isfinite(logits).all()
    
    info = get_model_info(model)
    assert "total_parameters" in info and isinstance(info["total_parameters"], int)
    assert "num_classes" in info and info["num_classes"] == 10
