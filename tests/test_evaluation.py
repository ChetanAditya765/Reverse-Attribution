"""
tests/test_evaluation.py

Unit-tests for evaluation framework using your actual model implementations and RA.
"""

import torch
import pytest
from ra.evaluate import ModelEvaluator, check_evaluation_compatibility
from models.bert_sentiment import create_bert_sentiment_model
from models.resnet_cifar import resnet56_cifar
from ra.dataset_utils import DatasetLoader
import numpy as np


@pytest.fixture(scope="module")
def bert_model_and_data():
    model = create_bert_sentiment_model("prajjwal1/bert-tiny", num_classes=2).eval()
    loader = DatasetLoader()
    dataloader = loader.create_text_dataloader(
        "imdb", "test", model.tokenizer, batch_size=4, shuffle=False
    )
    return model, dataloader


@pytest.fixture(scope="module")
def resnet_model_and_data():
    model = resnet56_cifar(num_classes=10).eval()
    loader = DatasetLoader()
    dataloader = loader.create_vision_dataloader(
        split="test", batch_size=4, shuffle=False
    )
    return model, dataloader


def test_bert_evaluation_metrics(bert_model_and_data):
    model, dl = bert_model_and_data
    evaluator = ModelEvaluator(model, device="cpu", save_dir="./tmp_eval")
    
    metrics = evaluator.evaluate_standard_metrics(dl, dataset_name="imdb")
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert "ece" in metrics and isinstance(metrics["ece"], float)
    assert metrics["model_type"] in {"bert_sentiment", "text_transformer"}
    
    ra_res = evaluator.evaluate_reverse_attribution(dl, "imdb", max_samples=5)
    assert "summary" in ra_res and "avg_a_flip" in ra_res["summary"]


def test_resnet_evaluation_metrics(resnet_model_and_data):
    model, dl = resnet_model_and_data
    evaluator = ModelEvaluator(model, device="cpu", save_dir="./tmp_eval")
    
    metrics = evaluator.evaluate_standard_metrics(dl, dataset_name="cifar10")
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["model_type"] in {"resnet_cifar", "vision_cnn"}
    
    ra_res = evaluator.evaluate_reverse_attribution(dl, "cifar10", max_samples=5)
    assert "summary" in ra_res and "avg_counter_evidence_count" in ra_res["summary"]


def test_evaluation_compatibility():
    status = check_evaluation_compatibility()
    # At least one model type should be available
    assert any(status.values())
