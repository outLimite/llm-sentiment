"""
DoRA (Weight-Decomposed Low-Rank Adaptation) implementation 
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
from .lora import LoRALayer
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

class LinearWithDoRA(nn.Module):
    """
    Combines a standard linear layer with DoRA adaptation.
    
    DoRA decomposes weight updates into magnitude and direction:
    W = m * (W0 + BA) / ||W0 + BA||
    
    Attributes:
        linear: Original linear layer
        lora: LoRA adaptation layer
        W0: Frozen original weights
        m: Learnable magnitude vector
    """
    
    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)
        
        self.register_buffer("W0", linear.weight.data.clone())
        
        self.m = nn.Parameter(torch.ones(linear.out_features, 1))
        
        logger.info(f"Initialized LinearWithDoRA for linear layer with {linear.in_features}->{linear.out_features}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Linear + DoRA.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        delta_W = self.lora.scaling * (self.lora.A.weight @ self.lora.B.weight)
        
        W_hat = self.W0 + delta_W
        
        norm = W_hat.norm(dim=1, keepdim=True) + 1e-8  # Add epsilon for numerical stability
        direction = W_hat / norm
        
        W = self.m * direction
        
        if self.linear.bias is not None:
            return F.linear(x, W, self.linear.bias)
        else:
            return F.linear(x, W)
        

def setup_dora_model(
    model: nn.Module,
    target_modules: List[str] = None,
    r: int = 8,
    alpha: int = 16
) -> nn.Module:
    """
    Quick setup for DoRA adaptation.
    
    Args:
        model: Base model to adapt
        target_modules: List of module names to apply DoRA to
        r: LoRA rank
        alpha: LoRA alpha
        
    Returns:
        Model with DoRA adaptation
    """
    if target_modules is None:
        target_modules = ["k_proj", "v_proj"] 
    
    from .base import apply_peft_to_module, freeze_layers
    
    apply_peft_to_module(model, LinearWithDoRA, r, alpha, target_modules)
    
    model = freeze_layers(model, ["lora", "m"])
    
    logger.info(f"Setup DoRA with r={r}, alpha={alpha}, target_modules={target_modules}")
    
    return model