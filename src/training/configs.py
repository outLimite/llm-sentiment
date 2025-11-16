"""
Training configurations and hyperparameters.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch


@dataclass
class TrainingConfig:
    """Configuration for PEFT training."""
    
    batch_size: int = 16
    learning_rate: float = 5e-4
    num_epochs: int = 3
    max_length: int = 256
    
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    
    lora_rank: int = 8
    lora_alpha: int = 16
    target_modules: tuple = ("k_proj", "v_proj")
    
    eval_batch_size: int = 100
    eval_steps: int = 50
    
    device: str = "auto"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate, 
            'num_epochs': self.num_epochs,
            'max_length': self.max_length,
            'weight_decay': self.weight_decay,
            'max_grad_norm': self.max_grad_norm,
            'lora_rank': self.lora_rank,
            'lora_alpha': self.lora_alpha,
            'target_modules': self.target_modules,
        }


def get_lora_config() -> TrainingConfig:
    """Get default configuration for LoRA training."""
    return TrainingConfig(
        batch_size=16,
        learning_rate=5e-4,
        num_epochs=3,
        lora_rank=8,
        lora_alpha=16,
        target_modules=("k_proj", "v_proj")
    )


def get_dora_config() -> TrainingConfig:
    """Get default configuration for DoRA training."""
    return TrainingConfig(
        batch_size=16, 
        learning_rate=5e-4,
        num_epochs=3,
        lora_rank=8,
        lora_alpha=16,
        target_modules=("k_proj", "v_proj")
    )


def get_qlora_config() -> TrainingConfig:
    """Get configuration for QLoRA training."""
    return TrainingConfig(
        batch_size=4, 
        learning_rate=2e-4,
        num_epochs=2,
        lora_rank=8,
        lora_alpha=16,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        gradient_accumulation_steps=4
    )