"""
Unit-tests for ReverseAttribution core logic
Run with:  pytest -q tests/test_ra_core.py
"""
import torch, pytest
from ra.model_factory import ModelFactory
from ra.ra import ReverseAttribution

@pytest.fixture(scope="module")
def tiny_text_model():
    return ModelFactory.create_text_model("prajjwal1/bert-tiny", num_classes=2)

def test_ra_outputs_dict(tiny_text_model):
    tok = tiny_text_model.tokenizer("Good film!", return_tensors="pt")
    ra  = ReverseAttribution(tiny_text_model, device="cpu")
    res = ra.explain(tok["input_ids"], y_true=1, top_m=3)
    required_keys = {"counter_evidence", "a_flip", "phi", "y_hat", "runner_up"}
    assert required_keys.issubset(res.keys())

def test_counter_evidence_non_positive(tiny_text_model):
    tok = tiny_text_model.tokenizer("Bad movie :(", return_tensors="pt")
    ra  = ReverseAttribution(tiny_text_model, device="cpu")
    res = ra.explain(tok["input_ids"], y_true=0, top_m=5)
    # ensure counter evidence has negative attribution values
    for _, phi, _ in res["counter_evidence"]:
        assert phi < 0
