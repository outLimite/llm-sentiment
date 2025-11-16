"""
LoRA (Low-Rank Adaptation) implementation
"""

import numpy as np 
import torch
import torch.nn as nn
from typing import List
import logging 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="logging/app.log",
    filemode="w",
    )
logger = logging.getLogger(__name__)

class LoRALayer(nn.Module):
    """Implements a low-rank adaptation layer for a linear transformation.
    This layer introduces a trainable low-rank update to the input tensor.

    The forward computation is defined as:
        output = alpha * (x @ A @ B)

    Attributes:
        B (nn.Parameter): A weight matrix of shape (in_dim, rank), initialized to zeros.
        A (nn.Parameter): A weight matrix of shape (rank, out_dim), initialized with random values
                            scaled by 1/sqrt(rank).
        alpha (float): A scaling factor for the low-rank update.
    """

    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float):
        super().__init__()
        self.A = nn.Linear(rank, out_dim, bias=False)
        self.B = nn.Linear(in_dim, rank, bias=False)

        self.scaling = alpha / rank

        self._initialize_weights()

        logger.debug(f"Initialized LoRALayer: in_dim={in_dim}, out_dim={out_dim}, rank={rank}, alpha={alpha}")

    def _initialize_weights(self):
        """Initialize LoRA weights according to the original paper."""
        nn.init.zeros_(self.B.weight)
        nn.init.normal_(self.A.weight, mean=0.0, std=1.0 / np.sqrt(self.rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LoRA layer.
        
        Args:
            x: Input tensor of shape (batch_size, in_dim)
            
        Returns:
            Output tensor of shape (batch_size, out_dim)
        """
        return self.scaling * self.A(self.B(x))

class LinearWithLoRA(nn.Module):
    """Combines a standard linear layer with a LoRA (Low-Rank Adaptation) layer.
    The forward pass returns the sum of the output of the linear layer and the low-rank update.

    Attributes:
        linear (nn.Module): The original linear layer.
        lora (LoRALayer): The low-rank adaptation layer configured with matching input and output dimensions.
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

        logger.debug(f"Initialized LinearWithLoRA for linear layer with {linear.in_features}->{linear.out_features}")

    def forward(self, x: torch.Tensor):
        """
        Forward pass through Linear + LoRA.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.linear(x) + self.lora(x)
    
def setup_lora_model(
    model: nn.Module,
    target_submodules: List[str] = None,
    r: int = 8,
    alpha: float = 16.0,
) -> nn.Module:
    """
    Quick setup for LoRA adaptation.
    
    Args:
        model: Base model to adapt
        target_modules: List of module names to apply LoRA to
        r: LoRA rank
        alpha: LoRA alpha
        
    Returns:
        Model with LoRA adaptation
    """
    if target_submodules is None:
        target_submodules = ["k_proj, v_proj"]

    from .base import apply_peft_to_model, freeze_layers

    apply_peft_to_model(model, LinearWithLoRA, r, alpha, target_submodules)

    model = freeze_layers(model, ["lora"])

    logger.info(f"Setup LoRA with r={r}, alpha={alpha}, target_submodules={target_submodules}")
    
    return model
     