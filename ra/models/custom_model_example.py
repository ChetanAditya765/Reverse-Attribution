"""
Example showing how to integrate custom models with the Reverse Attribution framework.
Demonstrates the required interfaces and best practices for RA compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from transformers import AutoTokenizer

# Import RA framework
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ra.ra import ReverseAttribution
from ra.model_factory import ModelFactory
from visualizer import ExplanationVisualizer


class CustomTextClassifier(nn.Module):
    """
    Example custom text classifier that works with Reverse Attribution.
    
    Key requirements for RA compatibility:
    1. Must be a PyTorch nn.Module
    2. Forward pass should return logits
    3. Should have an embeddings layer accessible for layer-wise attribution
    4. Input format should be consistent (token IDs, attention masks, etc.)
    """
    
    def __init__(
        self,
        vocab_size: int = 30522,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 2,
        max_length: int = 512,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.max_length = max_length
        
        # Embedding layer (required for RA layer attribution)
        self.embeddings = nn.ModuleDict({
            'word_embeddings': nn.Embedding(vocab_size, embedding_dim, padding_idx=0),
            'position_embeddings': nn.Embedding(max_length, embedding_dim),
            'layer_norm': nn.LayerNorm(embedding_dim),
            'dropout': nn.Dropout(dropout_rate)
        })
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass - must return logits for RA compatibility.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        word_embeds = self.embeddings['word_embeddings'](input_ids)
        
        # Add positional embeddings
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.embeddings['position_embeddings'](position_ids)
        
        # Combine embeddings
        embeddings = word_embeds + position_embeds
        embeddings = self.embeddings['layer_norm'](embeddings)
        embeddings = self.embeddings['dropout'](embeddings)
        
        # LSTM encoding
        lstm_out, (hidden, cell) = self.encoder[0](embeddings)
        
        # Use final hidden state for classification
        # Take the last hidden state from both directions
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [batch_size, hidden_dim*2]
        
        # Classification
        logits = self.classifier(final_hidden)
        
        return logits


class CustomVisionClassifier(nn.Module):
    """
    Example custom CNN classifier for image data.
    Demonstrates RA compatibility for vision models.
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 3,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for vision classifier.
        
        Args:
            x: Input images [batch_size, channels, height, width]
            
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        features = self.features(x)
        logits = self.classifier(features)
        return logits


class CustomModelWrapper:
    """
    Wrapper class to make custom models work seamlessly with RA framework.
    Provides standardized interfaces for different model types.
    """
    
    def __init__(self, model: nn.Module, tokenizer: Optional[Any] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
    
    def predict(self, inputs: Any) -> Dict[str, np.ndarray]:
        """Standardized prediction interface."""
        self.model.eval()
        
        with torch.no_grad():
            if isinstance(inputs, dict):
                # Text inputs
                logits = self.model(**inputs)
            else:
                # Image inputs or simple tensors
                logits = self.model(inputs)
            
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
        
        return {
            'predictions': preds.cpu().numpy(),
            'probabilities': probs.cpu().numpy(),
            'logits': logits.cpu().numpy()
        }
    
    def encode_text(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Encode text inputs (for text models)."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for text encoding")
        
        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )


def demonstrate_custom_text_model():
    """
    Demonstrate how to use a custom text model with Reverse Attribution.
    """
    print("🔤 Demonstrating Custom Text Model with RA")
    print("=" * 50)
    
    # Create custom model
    model = CustomTextClassifier(
        vocab_size=30522,  # BERT vocabulary size
        embedding_dim=128,
        hidden_dim=256,
        num_classes=2
    )
    
    # Load tokenizer for text processing
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Create wrapper
    wrapper = CustomModelWrapper(model, tokenizer)
    
    # Create RA explainer
    ra_explainer = ReverseAttribution(model, device="cpu")
    
    # Example text
    text = "This movie is not very good and quite boring."
    print(f"Input text: {text}")
    
    # Encode text
    encoded = wrapper.encode_text([text])
    print(f"Encoded shape: {encoded['input_ids'].shape}")
    
    # Make prediction
    prediction = wrapper.predict(encoded)
    pred_class = prediction['predictions'][0]
    confidence = prediction['probabilities'][0].max()
    
    print(f"Prediction: Class {pred_class} (confidence: {confidence:.3f})")
    
    # Generate RA explanation
    ra_result = ra_explainer.explain(
        encoded['input_ids'],
        y_true=1,  # Assume true label is positive
        top_m=5
    )
    
    print(f"\nRA Analysis:")
    print(f"A-Flip Score: {ra_result['a_flip']:.3f}")
    print(f"Counter-evidence features: {len(ra_result['counter_evidence'])}")
    
    if ra_result['counter_evidence']:
        print("Top counter-evidence:")
        for idx, (feat_idx, attribution, delta) in enumerate(ra_result['counter_evidence'][:3]):
            token = tokenizer.decode([encoded['input_ids'][0][feat_idx]])
            print(f"  {idx+1}. Token '{token}' (idx={feat_idx}): attr={attribution:.3f}, delta={delta:.3f}")
    
    return model, ra_result


def demonstrate_custom_vision_model():
    """
    Demonstrate how to use a custom vision model with Reverse Attribution.
    """
    print("\n🖼️ Demonstrating Custom Vision Model with RA")
    print("=" * 50)
    
    # Create custom vision model
    model = CustomVisionClassifier(num_classes=10, input_channels=3)
    
    # Create wrapper
    wrapper = CustomModelWrapper(model)
    
    # Create RA explainer
    ra_explainer = ReverseAttribution(model, device="cpu")
    
    # Generate sample image (32x32 RGB)
    sample_image = torch.randn(1, 3, 32, 32)
    print(f"Input image shape: {sample_image.shape}")
    
    # Make prediction
    prediction = wrapper.predict(sample_image)
    pred_class = prediction['predictions'][0]
    confidence = prediction['probabilities'][0].max()
    
    print(f"Prediction: Class {pred_class} (confidence: {confidence:.3f})")
    
    # Generate RA explanation
    ra_result = ra_explainer.explain(
        sample_image,
        y_true=5,  # Assume true label is class 5
        top_m=10
    )
    
    print(f"\nRA Analysis:")
    print(f"A-Flip Score: {ra_result['a_flip']:.3f}")
    print(f"Counter-evidence features: {len(ra_result['counter_evidence'])}")
    
    if ra_result['counter_evidence']:
        print("Counter-evidence pixel regions:")
        for idx, (feat_idx, attribution, delta) in enumerate(ra_result['counter_evidence'][:5]):
            # Convert flat index to 2D coordinates
            h, w = 32, 32  # Image dimensions
            y = feat_idx // (w * 3)  # Assuming CHW format flattened
            x = (feat_idx % (w * 3)) // 3
            channel = feat_idx % 3
            
            print(f"  {idx+1}. Pixel ({y}, {x}, ch={channel}): attr={attribution:.3f}, delta={delta:.3f}")
    
    return model, ra_result


def demonstrate_integration_best_practices():
    """
    Show best practices for integrating custom models with RA.
    """
    print("\n📋 Best Practices for RA Integration")
    print("=" * 50)
    
    practices = [
        "1. Model Output: Always return raw logits, not probabilities",
        "2. Embeddings: Expose embedding layers for layer-wise attribution",
        "3. Device Handling: Ensure model and inputs are on same device",
        "4. Input Format: Use consistent input format (dicts for text, tensors for vision)",
        "5. Batch Dimension: Always include batch dimension, even for single samples",
        "6. Gradient Flow: Ensure gradients flow through all relevant parameters",
        "7. Evaluation Mode: Set model to eval() mode during explanation generation",
        "8. Memory Management: Use torch.no_grad() when not computing explanations"
    ]
    
    for practice in practices:
        print(f"  ✅ {practice}")
    
    print(f"\n💡 Key Requirements:")
    print(f"  • Model must be a PyTorch nn.Module")
    print(f"  • Forward pass returns logits (not probabilities)")
    print(f"  • Embeddings accessible via model.embeddings (for text) or direct input (for vision)")
    print(f"  • Consistent input/output interfaces")


def run_complete_example():
    """
    Run complete example showing custom model integration with RA.
    """
    print("🚀 Complete Custom Model Integration Example")
    print("=" * 60)
    
    # Text model example
    text_model, text_ra_result = demonstrate_custom_text_model()
    
    # Vision model example
    vision_model, vision_ra_result = demonstrate_custom_vision_model()
    
    # Best practices
    demonstrate_integration_best_practices()
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"  • Text model A-Flip: {text_ra_result['a_flip']:.3f}")
    print(f"  • Vision model A-Flip: {vision_ra_result['a_flip']:.3f}")
    print(f"  • Both models successfully integrated with RA framework")
    
    return {
        'text_model': text_model,
        'vision_model': vision_model,
        'text_results': text_ra_result,
        'vision_results': vision_ra_result
    }


if __name__ == "__main__":
    # Run the complete demonstration
    results = run_complete_example()
    
    print(f"\n✅ Custom model integration demonstration completed!")
    print(f"Both text and vision models are now RA-compatible.")
