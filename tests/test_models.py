"""
PyTest unit-tests for model creation utilities.
Run with:  pytest -q tests/test_models.py
"""
import torch, pytest
from ra.model_factory import ModelFactory

@pytest.mark.parametrize("arch", ["bert-base-uncased", "roberta-base"])
def test_text_model_forward(arch):
    model = ModelFactory.create_text_model(arch, num_classes=2)
    tok   = model.tokenizer("Unit-test sentence.", return_tensors="pt")
    logits = model(tok["input_ids"], tok["attention_mask"])
    assert logits.shape == (1, 2)
    assert torch.isfinite(logits).all()

def test_vision_model_forward():
    model = ModelFactory.create_vision_model(architecture="resnet56", num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (2, 10)
    assert torch.isfinite(logits).all()
