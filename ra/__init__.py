"""
Reverse-Attribution public API
"""

from importlib.metadata import version

# sub-module public symbols
from .ra import ReverseAttribution
from .model_factory import ModelFactory
from .dataset_utils import DatasetLoader
from .explainer_utils import ExplainerHub
from . import metrics as metrics

__all__ = [
    "ReverseAttribution",
    "ModelFactory",
    "DatasetLoader",
    "ExplainerHub",
    "metrics",
    "__version__",
]

try:
    __version__ = version("reverse-attribution")
except Exception:  # package not yet installed
    __version__ = "0.0.0"
