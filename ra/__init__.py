"""
Reverse Attribution – public API

Exposes:
    • ReverseAttribution core explainer
    • ModelFactory aware of BERTSentimentClassifier & ResNetCIFAR
    • Dataset utilities
    • Evaluation framework
    • Metrics module
    • Visualization utilities
    • User-study components
"""

from importlib.metadata import version as _pkg_version

# Core algorithm
from .ra import ReverseAttribution

# Model factory
from .model_factory import ModelFactory

# Data handling
from .dataset_utils import DatasetLoader, TextDataset, get_dataset_info

# Evaluation & metrics
from .evaluate import (
    ModelEvaluator,
    evaluate_with_user_study_data,
    create_evaluation_report,
)
from . import metrics

# Visualization
from .visualizer import ExplanationVisualizer

# User studies
from .user_study import (
    UserStudySession,
    TrustCalibrationStudy,
    DebuggingTimeStudy,
    UserStudyAnalyzer,
)

# Convenience model shortcuts
def _safe_import(path: str, name: str):
    try:
        return __import__(path, fromlist=[name]).__dict__[name]
    except Exception:
        return None

get_bert_model     = _safe_import("models", "get_bert_model")
get_resnet56_model = _safe_import("models", "get_resnet56_model")

__all__ = [
    "ReverseAttribution",
    "ModelFactory",
    "get_bert_model",
    "get_resnet56_model",
    "DatasetLoader",
    "TextDataset",
    "get_dataset_info",
    "ModelEvaluator",
    "evaluate_with_user_study_data",
    "create_evaluation_report",
    "metrics",
    "ExplanationVisualizer",
    "UserStudySession",
    "TrustCalibrationStudy",
    "DebuggingTimeStudy",
    "UserStudyAnalyzer",
    "__version__",
]

try:
    __version__ = _pkg_version("reverse-attribution")
except Exception:
    __version__ = "0.0.0.dev"
