"""
tests/test_ra_core.py

Unit-tests for the core ReverseAttribution logic using your actual model implementations.
Verifies that:
- explain() returns correct keys and types
- counter_evidence tuples have negative attributions
- a_flip is non-negative
"""

import pytest
import torch
import numpy as np

from models.bert_sentiment import create_bert_sentiment_model
from models.resnet_cifar import resnet56_cifar
from ra.ra import ReverseAttribution

@pytest.fixture(scope="module")
def bert_tiny_model():
    # Use a tiny pretrained model for fast tests
    model = create_bert_sentiment_model("prajjwal1/bert-tiny", num_classes=2)
    model.eval()
    return model

@pytest.fixture(scope="module")
def resnet_model():
    model = resnet56_cifar(num_classes=10)
    model.eval()
    return model

def test_ra_output_keys_and_types_text(bert_tiny_model):
    tok = bert_tiny_model.tokenizer("Test sentence for RA.", return_tensors="pt")
    ra = ReverseAttribution(bert_tiny_model, device="cpu")
    res = ra.explain(tok["input_ids"], y_true=1, additional_forward_args=(tok["attention_mask"],), top_m=3)
    # Required keys
    assert set(res.keys()) >= {"counter_evidence", "a_flip", "phi", "y_hat", "runner_up", "model_type"}
    # Types
    assert isinstance(res["a_flip"], float)
    assert isinstance(res["phi"], np.ndarray)
    assert isinstance(res["counter_evidence"], list)
    assert res["model_type"] == "bert_sentiment"

def test_counter_evidence_negative_attributions_text(bert_tiny_model):
    tok = bert_tiny_model.tokenizer("Bad movie test.", return_tensors="pt")
    ra = ReverseAttribution(bert_tiny_model, device="cpu")
    res = ra.explain(tok["input_ids"], y_true=0, additional_forward_args=(tok["attention_mask"],), top_m=5)
    for idx, attr, delta in res["counter_evidence"]:
        assert attr < 0, f"Expected negative attribution, got {attr}"

def test_a_flip_non_negative_text(bert_tiny_model):
    tok = bert_tiny_model.tokenizer("Neutral test text.", return_tensors="pt")
    ra = ReverseAttribution(bert_tiny_model, device="cpu")
    res = ra.explain(tok["input_ids"], y_true=1, additional_forward_args=(tok["attention_mask"],))
    assert res["a_flip"] >= 0

def test_ra_output_keys_and_types_vision(resnet_model):
    x = torch.randn(1, 3, 32, 32)
    ra = ReverseAttribution(resnet_model, device="cpu")
    res = ra.explain(x, y_true=5, top_m=4)
    assert set(res.keys()) >= {"counter_evidence", "a_flip", "phi", "y_hat", "runner_up", "model_type"}
    assert isinstance(res["a_flip"], float)
    assert isinstance(res["phi"], np.ndarray)
    assert isinstance(res["counter_evidence"], list)
    assert res["model_type"] == "resnet_cifar" or res["model_type"] == "vision_cnn"

def test_counter_evidence_negative_attributions_vision(resnet_model):
    x = torch.randn(1, 3, 32, 32)
    ra = ReverseAttribution(resnet_model, device="cpu")
    res = ra.explain(x, y_true=2, top_m=5)
    for idx, attr, delta in res["counter_evidence"]:
        assert attr < 0, f"Vision CE attr should be negative, got {attr}"

def test_a_flip_non_negative_vision(resnet_model):
    x = torch.randn(1, 3, 32, 32)
    ra = ReverseAttribution(resnet_model, device="cpu")
    res = ra.explain(x, y_true=3)
    assert res["a_flip"] >= 0
