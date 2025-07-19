"""
Utility functions for model operations including loading, saving, 
preprocessing, and inference helpers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
import os
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import yaml


class ModelCheckpointManager:
    """
    Manages model checkpoints including saving, loading, and versioning.
    """
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        epoch: int = 0,
        metrics: Dict[str, float] = None,
        model_name: str = "model",
        is_best: bool = False,
        additional_info: Dict[str, Any] = None
    ) -> str:
        """
        Save model checkpoint with metadata.
        
        Args:
            model: PyTorch model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler
            epoch: Current epoch
            metrics: Performance metrics
            model_name: Name for the checkpoint
            is_best: Whether this is the best model so far
            additional_info: Any additional information to save
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'metrics': metrics or {},
            'model_architecture': str(model),
            'timestamp': torch.tensor([0]).item(),  # Simple timestamp placeholder
        }
        
        if optimizer:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
            
        if additional_info:
            checkpoint_data.update(additional_info)
        
        # Create model-specific directory
        model_dir = self.checkpoint_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        # Save regular checkpoint
        checkpoint_path = model_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint_data, checkpoint_path)
        
        # Save as best model if specified
        if is_best:
            best_path = model_dir / "best_model.pt"
            torch.save(checkpoint_data, best_path)
            
            # Save metadata separately for easy access
            metadata = {
                'epoch': epoch,
                'metrics': metrics,
                'checkpoint_path': str(checkpoint_path),
                'timestamp': checkpoint_data['timestamp']
            }
            
            with open(model_dir / "best_model_info.json", 'w') as f:
                json.dump(metadata, f, indent=2)
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            device: Device to load tensors to
            
        Returns:
            Checkpoint metadata
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if available
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'additional_info': {k: v for k, v in checkpoint.items() 
                              if k not in ['model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict']}
        }
    
    def get_best_checkpoint_path(self, model_name: str) -> Optional[str]:
        """Get path to best checkpoint for a model."""
        best_path = self.checkpoint_dir / model_name / "best_model.pt"
        return str(best_path) if best_path.exists() else None
    
    def list_checkpoints(self, model_name: str) -> List[str]:
        """List all checkpoints for a model."""
        model_dir = self.checkpoint_dir / model_name
        if not model_dir.exists():
            return []
        
        return [str(p) for p in model_dir.glob("checkpoint_epoch_*.pt")]


class ModelWrapper:
    """
    Generic wrapper for PyTorch models with common utilities.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = None,
        preprocessing_fn: callable = None,
        postprocessing_fn: callable = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.preprocessing_fn = preprocessing_fn
        self.postprocessing_fn = postprocessing_fn
        
    def predict(
        self,
        inputs: Union[torch.Tensor, np.ndarray, List],
        return_probabilities: bool = True,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Make predictions on inputs.
        
        Args:
            inputs: Input data
            return_probabilities: Whether to return probabilities or logits
            batch_size: Batch size for processing
            
        Returns:
            Predictions as numpy array
        """
        self.model.eval()
        
        # Preprocessing
        if self.preprocessing_fn:
            inputs = self.preprocessing_fn(inputs)
        
        # Convert to tensor if needed
        if not isinstance(inputs, torch.Tensor):
            if isinstance(inputs, np.ndarray):
                inputs = torch.from_numpy(inputs).float()
            elif isinstance(inputs, list):
                inputs = torch.tensor(inputs).float()
        
        inputs = inputs.to(self.device)
        
        # Process in batches
        all_outputs = []
        
        with torch.no_grad():
            for i in range(0, len(inputs), batch_size):
                batch = inputs[i:i+batch_size]
                outputs = self.model(batch)
                
                if return_probabilities:
                    outputs = F.softmax(outputs, dim=1)
                
                all_outputs.append(outputs.cpu())
        
        predictions = torch.cat(all_outputs, dim=0).numpy()
        
        # Postprocessing
        if self.postprocessing_fn:
            predictions = self.postprocessing_fn(predictions)
        
        return predictions
    
    def predict_single(self, input_sample: Any) -> Dict[str, Any]:
        """
        Make prediction on single sample with detailed output.
        
        Args:
            input_sample: Single input sample
            
        Returns:
            Dictionary with prediction details
        """
        if isinstance(input_sample, (list, np.ndarray)):
            input_tensor = torch.tensor([input_sample]).float().to(self.device)
        elif isinstance(input_sample, torch.Tensor):
            if input_sample.dim() == 1:
                input_tensor = input_sample.unsqueeze(0).to(self.device)
            else:
                input_tensor = input_sample.to(self.device)
        else:
            raise ValueError(f"Unsupported input type: {type(input_sample)}")
        
        self.model.eval()
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)
            
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = probs[0, predicted_class].item()
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': probs[0].cpu().numpy(),
                'logits': logits[0].cpu().numpy()
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the wrapped model."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_architecture': str(self.model),
            'device': str(self.device),
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }


class TextModelUtils:
    """
    Utilities specific to text models (BERT, RoBERTa, etc.).
    """
    
    @staticmethod
    def create_attention_mask(input_ids: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
        """Create attention mask from input_ids."""
        return (input_ids != pad_token_id).long()
    
    @staticmethod
    def truncate_sequences(
        input_ids: torch.Tensor,
        max_length: int,
        truncation_strategy: str = "longest_first"
    ) -> torch.Tensor:
        """
        Truncate sequences to maximum length.
        
        Args:
            input_ids: Input token IDs
            max_length: Maximum sequence length
            truncation_strategy: How to truncate ('longest_first', 'only_first', 'only_second')
            
        Returns:
            Truncated input_ids
        """
        if input_ids.size(1) <= max_length:
            return input_ids
        
        if truncation_strategy == "longest_first":
            return input_ids[:, :max_length]
        elif truncation_strategy == "only_first":
            # Assumes [CLS] token1 [SEP] token2 [SEP] format
            return input_ids[:, :max_length]
        else:
            raise ValueError(f"Unknown truncation strategy: {truncation_strategy}")
    
    @staticmethod
    def extract_embeddings(
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        layer_index: int = -1
    ) -> torch.Tensor:
        """
        Extract embeddings from transformer model.
        
        Args:
            model: Transformer model
            input_ids: Input token IDs
            attention_mask: Attention mask
            layer_index: Which layer to extract from (-1 for last layer)
            
        Returns:
            Embeddings tensor
        """
        model.eval()
        
        with torch.no_grad():
            if hasattr(model, 'bert'):  # BERT-based models
                if attention_mask is not None:
                    outputs = model.bert(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                else:
                    outputs = model.bert(input_ids, output_hidden_states=True)
                
                hidden_states = outputs.hidden_states
                return hidden_states[layer_index]
            
            elif hasattr(model, 'roberta'):  # RoBERTa-based models
                if attention_mask is not None:
                    outputs = model.roberta(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                else:
                    outputs = model.roberta(input_ids, output_hidden_states=True)
                
                hidden_states = outputs.hidden_states
                return hidden_states[layer_index]
            
            else:
                raise ValueError("Model type not supported for embedding extraction")
    
    @staticmethod
    def create_text_preprocessing_fn(
        tokenizer: AutoTokenizer,
        max_length: int = 512
    ) -> callable:
        """Create preprocessing function for text inputs."""
        
        def preprocess(texts):
            if isinstance(texts, str):
                texts = [texts]
            
            encoded = tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors="pt"
            )
            
            return encoded
        
        return preprocess


class VisionModelUtils:
    """
    Utilities specific to vision models.
    """
    
    @staticmethod
    def create_vision_preprocessing_fn(
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        size: Tuple[int, int] = (224, 224)
    ) -> callable:
        """Create preprocessing function for vision inputs."""
        
        def preprocess(images):
            if isinstance(images, np.ndarray):
                if images.ndim == 3:  # Single image
                    images = images[np.newaxis, ...]
                
                # Normalize
                images = images.astype(np.float32) / 255.0
                
                # Standardize
                for i in range(3):  # RGB channels
                    images[:, :, :, i] = (images[:, :, :, i] - mean[i]) / std[i]
                
                # Convert to torch tensor and change dimension order
                images = torch.from_numpy(images).permute(0, 3, 1, 2)
            
            return images
        
        return preprocess
    
    @staticmethod
    def extract_feature_maps(
        model: nn.Module,
        inputs: torch.Tensor,
        layer_name: str = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract feature maps from specific layers.
        
        Args:
            model: Vision model
            inputs: Input images
            layer_name: Specific layer name (None for all)
            
        Returns:
            Dictionary of feature maps
        """
        feature_maps = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                feature_maps[name] = output.detach().clone()
            return hook
        
        # Register hooks
        handles = []
        for name, module in model.named_modules():
            if layer_name is None or name == layer_name:
                handle = module.register_forward_hook(hook_fn(name))
                handles.append(handle)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            _ = model(inputs)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        return feature_maps


class ConfigManager:
    """
    Manages configuration files and hyperparameters.
    """
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str):
        """Save configuration to file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == '.json':
                json.dump(config, f, indent=2)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two configuration dictionaries."""
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigManager.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in megabytes."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def freeze_layers(model: nn.Module, layer_names: List[str]):
    """
    Freeze specific layers in a model.
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to freeze
    """
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = False
            print(f"Frozen layer: {name}")


def unfreeze_layers(model: nn.Module, layer_names: List[str]):
    """
    Unfreeze specific layers in a model.
    
    Args:
        model: PyTorch model  
        layer_names: List of layer names to unfreeze
    """
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = True
            print(f"Unfrozen layer: {name}")


def initialize_weights(model: nn.Module, init_type: str = "xavier"):
    """
    Initialize model weights.
    
    Args:
        model: PyTorch model
        init_type: Type of initialization ('xavier', 'kaiming', 'normal')
    """
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if init_type == "xavier":
                nn.init.xavier_uniform_(module.weight)
            elif init_type == "kaiming":
                nn.init.kaiming_uniform_(module.weight)
            elif init_type == "normal":
                nn.init.normal_(module.weight, 0, 0.01)
            
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


def create_model_summary(model: nn.Module) -> Dict[str, Any]:
    """
    Create comprehensive model summary.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model summary dictionary
    """
    param_info = count_parameters(model)
    size_mb = get_model_size_mb(model)
    
    # Layer breakdown
    layer_info = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            layer_params = sum(p.numel() for p in module.parameters())
            layer_info.append({
                'name': name,
                'type': type(module).__name__,
                'parameters': layer_params
            })
    
    return {
        'total_parameters': param_info['total_parameters'],
        'trainable_parameters': param_info['trainable_parameters'],
        'model_size_mb': size_mb,
        'num_layers': len(layer_info),
        'layer_breakdown': layer_info
    }


class InferenceOptimizer:
    """
    Utilities for optimizing model inference.
    """
    
    @staticmethod
    def convert_to_half_precision(model: nn.Module) -> nn.Module:
        """Convert model to half precision (FP16)."""
        return model.half()
    
    @staticmethod
    def enable_torch_compile(model: nn.Module) -> nn.Module:
        """Enable PyTorch 2.0 compilation if available."""
        try:
            import torch._dynamo
            return torch.compile(model)
        except ImportError:
            print("torch.compile not available, skipping compilation")
            return model
    
    @staticmethod
    def optimize_for_inference(
        model: nn.Module,
        use_half_precision: bool = False,
        use_compile: bool = False
    ) -> nn.Module:
        """Apply various inference optimizations."""
        model.eval()
        
        if use_half_precision:
            model = InferenceOptimizer.convert_to_half_precision(model)
        
        if use_compile:
            model = InferenceOptimizer.enable_torch_compile(model)
        
        return model


if __name__ == "__main__":
    # Example usage
    
    # Test checkpoint manager
    print("Testing checkpoint manager...")
    checkpoint_manager = ModelCheckpointManager("./test_checkpoints")
    
    # Create dummy model
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    
    # Test saving
    save_path = checkpoint_manager.save_checkpoint(
        model=model,
        epoch=1,
        metrics={'accuracy': 0.85, 'loss': 0.3},
        model_name="test_model",
        is_best=True
    )
    print(f"Checkpoint saved to: {save_path}")
    
    # Test loading
    loaded_info = checkpoint_manager.load_checkpoint(save_path, model)
    print(f"Loaded checkpoint from epoch: {loaded_info['epoch']}")
    print(f"Loaded metrics: {loaded_info['metrics']}")
    
    # Test model wrapper
    print("\nTesting model wrapper...")
    wrapper = ModelWrapper(model)
    
    # Test prediction
    test_input = torch.randn(5, 10)
    predictions = wrapper.predict(test_input)
    print(f"Prediction shape: {predictions.shape}")
    
    # Test single prediction
    single_pred = wrapper.predict_single(test_input[0])
    print(f"Single prediction: class={single_pred['predicted_class']}, confidence={single_pred['confidence']:.3f}")
    
    # Test model info
    model_info = wrapper.get_model_info()
    print(f"Model info: {model_info}")
    
    # Test model summary
    print("\nTesting model summary...")
    summary = create_model_summary(model)
    print(f"Model summary: {summary}")
    
    print("\nAll tests passed!")
