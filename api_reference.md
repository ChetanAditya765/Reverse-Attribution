# API Reference

This document provides a reference for the core components of the Reverse Attribution library.

---

## Core Explainer

### `ra.ReverseAttribution`

The main class for running the Reverse Attribution algorithm.

```python
class ReverseAttribution:
    def __init__(self, model: nn.Module, baseline: Tensor|None=None, device: str=None):
        """Initializes the explainer."""
        pass

    def explain(
        self,
        x: Tensor,
        y_true: int,
        top_m: int = 5,
        n_steps: int = 50,
        additional_forward_args: tuple|None = None
    ) -> dict:
        """
        Generates the reverse attribution explanation for a given input.

        Returns:
            A dictionary with the following keys:
            - `counter_evidence`: The identified counter-evidence features.
            - `a_flip`: The A-Flip instability score.
            - `phi`: The final attribution scores.
            - `y_hat`: The model's predicted class.
            - `runner_up`: The second most likely class.
            - `model_type`: The type of the model ('text' or 'vision').
        """
        pass
```

---

## Model Factory

### `ra.model_factory.ModelFactory`

A factory for creating pre-configured text and vision models.

```python
class ModelFactory:
    @staticmethod
    def create_text_model(model_name, num_classes, checkpoint_path=None, model_type="bert_sentiment"):
        """Creates and returns a text-based model."""
        pass

    @staticmethod
    def create_vision_model(architecture, num_classes, checkpoint_path=None, model_type="resnet_cifar"):
        """Creates and returns a vision-based model."""
        pass
```

---

## Dataset Utilities

### `ra.dataset_utils.DatasetLoader`

Handles loading and preparing datasets.

```python
class DatasetLoader:
    def __init__(self, data_dir):
        """Initializes the dataset loader."""
        pass

    def create_text_dataloader(self, dataset_name, split, tokenizer, batch_size, max_length, shuffle):
        """Creates a DataLoader for text datasets."""
        pass

    def create_vision_dataloader(self, split, batch_size, shuffle):
        """Creates a DataLoader for vision datasets."""
        pass

def get_dataset_info(dataset_name) -> dict:
    """Returns metadata for a given dataset."""
    pass
```

---

## Evaluation

### `ra.evaluate.ModelEvaluator`

Provides tools for evaluating model performance and explanations.

```python
class ModelEvaluator:
    def __init__(self, model, device=None, save_dir="./results"):
        """Initializes the evaluator."""
        pass

    def evaluate_standard_metrics(self, dataloader, dataset_name):
        """Evaluates the model on standard performance metrics."""
        pass

    def evaluate_reverse_attribution(self, dataloader, dataset_name, max_samples, focus_on_errors, top_m):
        """Runs a full evaluation using Reverse Attribution."""
        pass

    def evaluate_comprehensive(self, dataloader, dataset_name, ra_config):
        """Runs a comprehensive evaluation including baselines."""
        pass
```

---

## Baseline Explainers

### `ra.explainer_utils.ExplainerHub`

A centralized hub for accessing and running baseline explanation methods like SHAP, LIME, and Captum.

```python
class ExplainerHub:
    def __init__(self, model, device):
        """Initializes the hub with a model."""
        pass

    def get_available_explainers(self) -> list:
        """Returns a list of available explainer methods (e.g., ["shap", "lime", "captum"])."""
        pass

    def explain_with_all(self, input_data, target_class):
        """Generates explanations using all available methods."""
        pass

    def explain_with_method(self, method_name, input_data, target_class):
        """Generates an explanation using a specific method."""
        pass
```

---

## Visualization

### `ra.visualizer.ExplanationVisualizer`

Utilities for visualizing explanations.

```python
class ExplanationVisualizer:
    def __init__(self, save_dir):
        """Initializes the visualizer."""
        pass

    def visualize_text_explanation(self, tokens, attributions, title, save_name):
        """Visualizes attributions for a text input."""
        pass

    def visualize_image_explanation(self, image_tensor, attributions, title, save_name):
        """Visualizes attributions for an image input."""
        pass

    def visualize_ra_explanation(self, ra_results, input_data=None, tokenizer=None):
        """Creates a comprehensive visualization for a Reverse Attribution result."""
        pass
```

---

## User Studies

### `ra.user_study.UserStudySession`

A class for managing and recording data from user study sessions.

```python
class UserStudySession:
    @staticmethod
    def new_session(participant_id, study_name, out_dir):
        """Creates a new user study session."""
        pass

    def record_trust(self, sample_id, condition, before, after, meta):
        """Records trust calibration data."""
        pass

    def record_debug_time(self, sample_id, condition, seconds, success, meta):
        """Records debugging time and success."""
        pass

    def save(self):
        """Saves the session data to a file."""
        pass
