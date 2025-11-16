"""
Training utilities - padding, collate functions, etc.
"""

import torch
import random
import numpy as np
from typing import List, Dict, Any
from functools import partial
import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    logger.info(f"Set random seed to {seed}")


def pad_tensors(
    tensors: List[torch.Tensor], 
    padding_value: int = 0, 
    padding_side: str = "left"
) -> torch.Tensor:
    """
    Pad a list of tensors to the same size along their leading dimension.
    
    Args:
        tensors: List of tensors to pad
        padding_value: Value used for padding
        padding_side: Side to apply padding ("left" or "right")
        
    Returns:
        Padded tensor stack
    """
    if not tensors:
        raise ValueError("List of tensors is empty")

    max_length = max(t.size(0) for t in tensors)
    padded_tensors = []
    
    for t in tensors:
        pad_len = max_length - t.size(0)
        if pad_len > 0:
            pad_tensor = torch.full(
                (pad_len,), 
                padding_value, 
                dtype=t.dtype, 
                device=t.device
            )
            if padding_side == "left":
                t = torch.cat([pad_tensor, t], dim=0)
            elif padding_side == "right":
                t = torch.cat([t, pad_tensor], dim=0)
            else:
                raise ValueError("padding_side must be either 'left' or 'right'")
        
        padded_tensors.append(t)

    return torch.stack(padded_tensors)


def create_collate_fn(pad_token_id: int, padding_side: str = "right"):
    """
    Create collate function for DataLoader.
    
    Args:
        pad_token_id: Token ID for padding
        padding_side: Side to apply padding
        
    Returns:
        Collate function
    """
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate and pad a batch of tokenized examples for causal LM training.
        
        Each example should contain "full_input_ids" for the full sequence.
        """
        full_input_ids = [
            torch.tensor(ex["full_input_ids"], dtype=torch.long) 
            for ex in batch 
            if "full_input_ids" in ex
        ]
        
        if not full_input_ids:
            raise ValueError("No full_input_ids found in batch")
        
        padded_input_ids = pad_tensors(
            full_input_ids, 
            padding_value=pad_token_id, 
            padding_side=padding_side
        )
        
        attention_mask = (padded_input_ids != pad_token_id).long()
        
        return {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "labels": padded_input_ids, 
        }
    
    return collate_fn


def create_sft_collate_fn(tokenizer, max_length: int = 256):
    """
    Create collate function for SFT training with conversational format.
    
    Args:
        tokenizer: Tokenizer for encoding
        max_length: Maximum sequence length
        
    Returns:
        Collate function for SFT
    """
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function for SFT with messages format."""
        texts = [ex["text"] for ex in batch]
        
        encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": encodings["input_ids"].clone(), 
        }
    
    return collate_fn


def get_optimizer(model, learning_rate: float, weight_decay: float = 0.01):
    """
    Create optimizer for PEFT training.
    
    Args:
        model: Model with parameters
        learning_rate: Learning rate
        weight_decay: Weight decay for optimizer
        
    Returns:
        Configured optimizer
    """
    trainable_params = [
        p for n, p in model.named_parameters() 
        if p.requires_grad
    ]
    
    logger.info(f"Optimizing {len(trainable_params)} parameter groups")
    
    return torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )


def get_lr_scheduler(optimizer, num_training_steps: int, warmup_steps: int = 100):
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        num_training_steps: Total number of training steps
        warmup_steps: Number of warmup steps
        
    Returns:
        LR scheduler
    """
    from transformers import get_linear_schedule_with_warmup
    
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )



"""
Postprocessing utilities for sentiment analysis.
"""

import re
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SentimentPostprocessor:
    """
    Postprocess generated text to extract sentiment labels.
    """
    
    def __init__(self):
        self.positive_patterns = [
            r'\bpositive\b',
            r'\bpos\b', 
            r'\bgood\b',
            r'\bhappy\b',
            r'\bgreat\b',
            r'\bexcellent\b',
            r'\bamazing\b'
        ]
        
        self.negative_patterns = [
            r'\bnegative\b',
            r'\bneg\b',
            r'\bbad\b',
            r'\bsad\b',
            r'\bterrible\b',
            r'\bawful\b',
            r'\bhorrible\b'
        ]
        
        self.neutral_patterns = [
            r'\bneutral\b',
            r'\bneut\b',
            r'\bnormal\b',
            r'\bmedium\b',
            r'\baverage\b',
            r'\bok\b',
            r'\bneutral\b'
        ]
        
        # Compiled regex patterns for better performance
        self.positive_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.positive_patterns]
        self.negative_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.negative_patterns]
        self.neutral_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.neutral_patterns]

    def extract_sentiment(self, text: str) -> str:
        """
        Extract sentiment label from generated text.
        
        Args:
            text: Generated text from model
            
        Returns:
            Sentiment label: "positive", "negative", "neutral", or ""
        """
        if not text or not text.strip():
            return ""
            
        text_lower = text.lower().strip()
        
        # Check for exact matches first
        if any(pattern.search(text_lower) for pattern in self.positive_regex):
            return "positive"
        elif any(pattern.search(text_lower) for pattern in self.negative_regex):
            return "negative"
        elif any(pattern.search(text_lower) for pattern in self.neutral_regex):
            return "neutral"
        
        # Fallback: check for sentiment indicators
        return self._fallback_sentiment_analysis(text_lower)

    def _fallback_sentiment_analysis(self, text: str) -> str:
        """
        Fallback method for sentiment analysis using keyword matching.
        """
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'like', 'happy', 'positive']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'negative', 'worst']
        neutral_words = ['neutral', 'ok', 'fine', 'average', 'normal', 'medium']
        
        words = text.split()
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        neutral_count = sum(1 for word in words if word in neutral_words)
        
        if positive_count > negative_count and positive_count > neutral_count:
            return "positive"
        elif negative_count > positive_count and negative_count > neutral_count:
            return "negative"
        elif neutral_count > positive_count and neutral_count > negative_count:
            return "neutral"
        else:
            return ""

    def batch_extract_sentiment(self, texts: List[str]) -> List[str]:
        """
        Extract sentiment labels for multiple texts.
        
        Args:
            texts: List of generated texts
            
        Returns:
            List of sentiment labels
        """
        return [self.extract_sentiment(text) for text in texts]

    def get_sentiment_confidence(self, text: str) -> Dict[str, Any]:
        """
        Get sentiment with confidence score.
        
        Args:
            text: Generated text
            
        Returns:
            Dictionary with sentiment and confidence
        """
        sentiment = self.extract_sentiment(text)
        
        if sentiment:
            return {
                "sentiment": sentiment,
                "confidence": 1.0,
                "raw_text": text
            }
        else:
            return {
                "sentiment": "",
                "confidence": 0.0,
                "raw_text": text
            }
