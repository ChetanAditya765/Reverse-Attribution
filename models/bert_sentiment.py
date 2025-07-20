"""
BERT-based sentiment analysis model with RA integration.
Provides a clean interface for sentiment classification tasks on IMDB, Yelp, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class BERTSentimentClassifier(nn.Module):
    """
    BERT-based sentiment classifier with additional features for RA analysis.
    Supports fine-tuning and provides intermediate representations.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 2,
        dropout_rate: float = 0.1,
        freeze_bert: bool = False,
        use_pooler: bool = True
    ):
        super().__init__()

        # Import device after initialization
        from ra.device_utils import device
        self.device = device
    
        self.model_name = model_name
        self.num_classes = num_classes
        self.use_pooler = use_pooler

        # Load BERT configuration and model
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
        # Move model to GPU
        self.to(self.device)

        # Freeze BERT parameters if requested
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Classification head
        hidden_size = self.config.hidden_size
        
        if use_pooler:
            # Use BERT's pooler output
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size // 2, num_classes)
            )
        else:
            # Use [CLS] token representation
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, num_classes)
            )
        
        # Initialize classifier weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the model.
        """
        # Move inputs to device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            return_hidden_states: Whether to return hidden states
            
        Returns:
            logits: Classification logits [batch_size, num_classes]
            hidden_states: (optional) Hidden states from BERT
        """
        # BERT forward pass
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=return_hidden_states
        )
        
        # Get representation for classification
        if self.use_pooler:
            # Use pooler output (already processed [CLS] token)
            pooled_output = bert_outputs.pooler_output
        else:
            # Use raw [CLS] token representation
            pooled_output = bert_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Classification
        logits = self.classifier(pooled_output)
        
        if return_hidden_states:
            return logits, bert_outputs.hidden_states
        return logits
    
    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings for RA analysis."""
        return self.bert.embeddings.word_embeddings(input_ids)
    
    def get_attention_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get attention weights for analysis."""
        with torch.no_grad():
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
            # Average over all heads and layers
            attentions = torch.stack(outputs.attentions)  # [layers, batch, heads, seq, seq]
            return attentions.mean(dim=(0, 2))  # [batch, seq, seq]
    
    def encode_text(
        self,
        texts: Union[str, List[str]],
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text inputs for model consumption.
        
        Args:
            texts: Input text(s)
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate long sequences
            
        Returns:
            Dictionary with input_ids, attention_mask, etc.
        """
        if isinstance(texts, str):
            texts = [texts]
        
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt"
        )
        
        return encoded
    
    def predict(
        self,
        texts: Union[str, List[str]],
        return_probabilities: bool = True,
        device: str = "cpu"
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions on text inputs.
        
        Args:
            texts: Input text(s)
            return_probabilities: Whether to return probabilities
            device: Device for computation
            
        Returns:
            Dictionary with predictions and probabilities
        """
        self.eval()
        
        # Encode texts
        encoded = self.encode_text(texts)
        
        # Move to device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        with torch.no_grad():
            logits = self.forward(encoded['input_ids'], encoded['attention_mask'])
            
            if return_probabilities:
                probs = F.softmax(logits, dim=1)
            else:
                probs = logits
            
            predictions = torch.argmax(logits, dim=1)
        
        return {
            'predictions': predictions.cpu().numpy(),
            'probabilities': probs.cpu().numpy(),
            'logits': logits.cpu().numpy()
        }
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model information for debugging."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'vocab_size': self.config.vocab_size,
            'hidden_size': self.config.hidden_size,
            'num_attention_heads': self.config.num_attention_heads,
            'num_hidden_layers': self.config.num_hidden_layers
        }


class BERTSentimentTrainer:
    """
    Training wrapper for BERT sentiment classification.
    """
    
    def __init__(
        self,
        model: BERTSentimentClassifier,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 0
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: str = None
    ) -> Dict[str, float]:
        """Train for one epoch."""
        from ra.device_utils import device as auto_device
        device = device or auto_device
    
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
    
        
        for batch in dataloader:
        # Move batch to device
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: str = None
    ) -> Dict[str, float]:
        """Evaluate the model."""
        from ra.device_utils import device as auto_device
        device = device or auto_device
    
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
    
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)

                
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }


def create_bert_sentiment_model(
    model_name: str = "bert-base-uncased",
    num_classes: int = 2,
    **kwargs
) -> BERTSentimentClassifier:
    """
    Factory function to create BERT sentiment model.
    
    Args:
        model_name: HuggingFace model name
        num_classes: Number of output classes
        **kwargs: Additional arguments for BERTSentimentClassifier
        
    Returns:
        Initialized BERTSentimentClassifier
    """
    return BERTSentimentClassifier(
        model_name=model_name,
        num_classes=num_classes,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    print("Creating BERT sentiment model...")
    
    model = create_bert_sentiment_model()
    print(f"Model info: {model.get_model_info()}")
    
    # Test prediction
    sample_texts = [
        "This movie is absolutely fantastic!",
        "I hate this boring film."
    ]
    
    results = model.predict(sample_texts)
    
    for i, text in enumerate(sample_texts):
        pred = results['predictions'][i]
        prob = results['probabilities'][i].max()
        sentiment = "Positive" if pred == 1 else "Negative"
        print(f"Text: {text}")
        print(f"Prediction: {sentiment} (confidence: {prob:.3f})")
        print()
