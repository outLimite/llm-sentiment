"""
Base utilities for PEFT model adaptation
"""

import torch
import torch.nn as nn
import numpy as np 
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

def setup_device():
    """
    Setup device for training (CPU, CUDA or MPS)
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logger.info(f"Using device: {device}")
    
    return device 

def apply_peft_to_model(
    model: nn.Module,
    adapter_class: nn.Module,
    r: int,
    alpha: float,
    target_submodules: List[str]
):
    """Recursively applies a parameter-efficient fine-tuning (PEFT) adapter to target submodules within a model.

    This function traverses the model's children recursively. For each submodule whose name contains any
    of the strings specified in 'target_submodules', it wraps the submodule using the provided adapter class.

    Args:
        model: The neural network model to modify.
        adapter_class: The adapter class (e.g., LoRALayer or LinearWithLoRA) used to wrap target submodules.
        r: The rank parameter for the adapter.
        alpha: The scaling factor for the low-rank update.
        target_submodules: A list of substrings to match against submodule names for applying the adapter.

    Returns:
        None. The function updates the model in-place.
    """
    replacements = 0 

    for name, module in model.named_modules():
        if any(t in name for t in target_submodules):
            parent_name = ".".join(name.split(".")[:-1])
            attr_name = name.split(".")[-1]

            try:
                parent = model.get_submodule(parent_name)
                origin_layer = getattr(parent, attr_name)

                setattr(parent, attr_name, adapter_class(origin_layer, r, alpha))
                replacements += 1

                logging.info(f"Replaced {name} with {adapter_class.__name__}")

            except Exception as e:
                logger.warning(f"Failed to replace {name}: {e}")

    logger.info(f"Applied {adapter_class.__name__} to {replacements} layers")

def freeze_layers(
    model: nn.Module,
    patterns: List[str]
) -> nn.Module: 
    """
    Freeze all layers except those matching the specified patterns.
    
    Args:
        model: Model to freeze layers for
        patterns: List of patterns for layers to keep trainable
        
    Returns:
        Model with frozen layers
    """
    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        if any(pat in name for pat in patterns):
            param.requires_grad = True
            trainable_params += np.prod(param.shape)
        else:
            param.requires_grad = False

        total_params += np.prod(param.shape)
    
    trainable_percent = (trainable_params / total_params * 100 if total_params > 0 else 0)
    logger.info(f"Trainable parameters: {trainable_params}/{total_params} ({trainable_percent:.2f}%)")

    return model 
    