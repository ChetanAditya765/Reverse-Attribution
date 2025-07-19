"""
Reverse Attribution - A framework for explaining model uncertainty via counter-evidence analysis.

This package provides tools for:
- Generating reverse attribution explanations
- Training and evaluating models
- Visualizing explanations and results
- Conducting user studies
- Comparing with baseline explanation methods
"""

from importlib.metadata import version

# Core RA algorithm
from .ra import ReverseAttribution

# Model utilities
from .model_factory import ModelFactory
from .model_utils import (
    ModelCheckpointManager,
    ModelWrapper,
    ConfigManager,
    count_parameters,
    get_model_size_mb,
    freeze_layers,
    unfreeze_layers
)

# Data handling
from .dataset_utils import DatasetLoader, TextDataset, get_dataset_info

# Evaluation and metrics
from .evaluate import ModelEvaluator, evaluate_with_user_study_data
from .metrics import (
    expected_calibration_error,
    jaccard_index,
    f1_score_localization,
    trust_change,
    compute_brier_score,
    evaluate_all_jmlr_metrics
)

# Explanation methods
from .explainer_utils import (
    ExplainerHub,
    BaselineExplainer,
    SHAPExplainer,
    LIMEExplainer,
    CaptumExplainer,
    normalize_attributions,
    extract_top_features
)

# Visualization
from .visualizer import (
    ExplanationVisualizer,
    create_word_cloud_from_attributions,
    save_explanation_report
)

# User studies
from .user_study import (
    UserStudySession,
    TrustCalibrationStudy,
    DebuggingTimeStudy,
    UserStudyAnalyzer
)

# Import metrics module for easier access
from . import metrics

__all__ = [
    # Core functionality
    "ReverseAttribution",
    
    # Model utilities
    "ModelFactory",
    "ModelCheckpointManager", 
    "ModelWrapper",
    "ConfigManager",
    "count_parameters",
    "get_model_size_mb",
    "freeze_layers",
    "unfreeze_layers",
    
    # Data handling
    "DatasetLoader",
    "TextDataset", 
    "get_dataset_info",
    
    # Evaluation
    "ModelEvaluator",
    "evaluate_with_user_study_data",
    
    # Metrics
    "expected_calibration_error",
    "jaccard_index",
    "f1_score_localization", 
    "trust_change",
    "compute_brier_score",
    "evaluate_all_jmlr_metrics",
    "metrics",
    
    # Explanation methods
    "ExplainerHub",
    "BaselineExplainer",
    "SHAPExplainer", 
    "LIMEExplainer",
    "CaptumExplainer",
    "normalize_attributions",
    "extract_top_features",
    
    # Visualization
    "ExplanationVisualizer",
    "create_word_cloud_from_attributions",
    "save_explanation_report",
    
    # User studies
    "UserStudySession",
    "TrustCalibrationStudy",
    "DebuggingTimeStudy", 
    "UserStudyAnalyzer",
    
    # Version
    "__version__",
]

try:
    __version__ = version("reverse-attribution")
except Exception:
    __version__ = "1.0.0"

# Package metadata
__author__ = "Chetan Aditya Lakka"
__email__ = "your.email@domain.com"
__description__ = "Reverse Attribution: Explaining Model Uncertainty via Counter-Evidence Analysis"
