"""
tests/test_integration.py

Integration tests to verify that the Reverse Attribution framework works
seamlessly with your actual model implementations:
- BERTSentimentClassifier
- ResNetCIFAR
- RA explanations
"""

import pytest
import torch
import numpy as np
from pathlib import Path

# Import your models
from models.bert_sentiment import BERTSentimentClassifier, create_bert_sentiment_model
from models.resnet_cifar import resnet56_cifar, get_model_info
from models.custom_model_example import CustomTextClassifier, CustomVisionClassifier

# Import RA framework
from ra.ra import ReverseAttribution


@pytest.fixture(scope="module")
def bert_model():
    # small vocab test, num_classes=2
    model = create_bert_sentiment_model("prajjwal1/bert-tiny", num_classes=2)
    return model.eval()

@pytest.fixture(scope="module")
def resnet_model():
    model = resnet56_cifar(num_classes=10)
    return model.eval()

@pytest.fixture(scope="module")
def custom_text_model():
    return CustomTextClassifier(vocab_size=1000, num_classes=2)

@pytest.fixture(scope="module")
def custom_vision_model():
    return CustomVisionClassifier(num_classes=5)

def test_bert_forward_and_ra(bert_model):
    # Dummy token IDs and attention mask
    input_ids = torch.randint(0, bert_model.tokenizer.vocab_size, (1, 16))
    attention_mask = torch.ones_like(input_ids)
    # Forward pass
    logits = bert_model(input_ids, attention_mask)
    assert logits.shape == (1, 2)
    assert torch.isfinite(logits).all()

    # RA explanation
    ra = ReverseAttribution(bert_model, device="cpu")
    res = ra.explain(input_ids, y_true=1, additional_forward_args=(attention_mask,))
    assert "a_flip" in res and isinstance(res["a_flip"], float)
    assert "phi" in res and isinstance(res["phi"], np.ndarray)
    assert "counter_evidence" in res and isinstance(res["counter_evidence"], list)
    assert res["model_type"] in {"bert_sentiment", "text_transformer"}

def test_resnet_forward_and_ra(resnet_model):
    # Dummy CIFAR image
    x = torch.randn(1, 3, 32, 32)
    logits = resnet_model(x)
    assert logits.shape == (1, 10)
    assert torch.isfinite(logits).all()

    # RA explanation
    ra = ReverseAttribution(resnet_model, device="cpu")
    res = ra.explain(x, y_true=3)
    assert "a_flip" in res and isinstance(res["a_flip"], float)
    assert "phi" in res and isinstance(res["phi"], np.ndarray)
    assert res["model_type"] in {"resnet_cifar", "vision_cnn"}

def test_custom_text_model_forward_and_ra(custom_text_model):
    # Dummy input IDs
    input_ids = torch.randint(0, custom_text_model.vocab_size, (1, 12))
    logits = custom_text_model(input_ids)
    assert logits.shape == (1, custom_text_model.num_classes)
    assert torch.isfinite(logits).all()

    ra = ReverseAttribution(custom_text_model, device="cpu")
    res = ra.explain(input_ids, y_true=0)
    assert res["model_type"] == "custom_text"
    assert isinstance(res["a_flip"], float)
    assert isinstance(res["phi"], np.ndarray)

def test_custom_vision_model_forward_and_ra(custom_vision_model):
    # Dummy image
    img = torch.randn(1, custom_vision_model.input_channels, 28, 28)
    logits = custom_vision_model(img)
    assert logits.shape == (1, custom_vision_model.num_classes)
    assert torch.isfinite(logits).all()

    ra = ReverseAttribution(custom_vision_model, device="cpu")
    res = ra.explain(img, y_true=1)
    assert res["model_type"] == "custom_vision"
    assert isinstance(res["a_flip"], float)
    assert isinstance(res["phi"], np.ndarray)
