"""
PEFT trainer implementation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Dict, Any, Optional
import logging

from .evaluator import evaluate_model
from .utils import get_optimizer, get_lr_scheduler

logger = logging.getLogger(__name__)


class PEFTTrainer:
    """
    Trainer for Parameter-Efficient Fine-Tuning.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataset,
        tokenizer,
        config
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.config = config
        
        self.optimizer = get_optimizer(
            model, 
            config.learning_rate, 
            config.weight_decay
        )
        
        self.scheduler = get_lr_scheduler(
            self.optimizer,
            num_training_steps=len(train_dataloader) * config.num_epochs
        )
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_f1 = 0.0
        
        logger.info(f"Initialized PEFTTrainer with {len(train_dataloader)} batches per epoch")

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(
            self.train_dataloader, 
            desc=f"Epoch {self.current_epoch + 1}",
            leave=False
        )
        
        for step, batch in enumerate(progress_bar):
            batch = self._prepare_batch(batch)
            
            outputs = self.model(**batch)
            loss = outputs.loss
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.max_grad_norm
            )
            
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            total_loss += loss.item()
            self.global_step += 1
            
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            if (step + 1) % self.config.eval_steps == 0:
                avg_loss = total_loss / (step + 1)
                logger.info(
                    f"Epoch {self.current_epoch + 1}, Step {step + 1}: "
                    f"Loss = {avg_loss:.4f}, LR = {self.scheduler.get_last_lr()[0]:.2e}"
                )
        
        epoch_loss = total_loss / len(self.train_dataloader)
        return epoch_loss

    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare batch for training by moving to device."""
        return {
            k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def evaluate(self) -> float:
        """Evaluate model on validation set."""
        logger.info("Running evaluation...")
        f1_score = evaluate_model(
            self.model, 
            self.val_dataset, 
            self.tokenizer,
            batch_size=self.config.eval_batch_size,
            show_conf_m=False
        )
        return f1_score

    def train(self) -> nn.Module:
        """Full training loop."""
        logger.info("Starting training...")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            train_loss = self.train_epoch()
            
            val_f1 = self.evaluate()
            
            logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val F1: {val_f1:.4f}"
            )
            
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                logger.info(f"New best F1: {val_f1:.4f}")
        
        logger.info(f"Training completed. Best F1: {self.best_f1:.4f}")
        return self.model

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_f1': self.best_f1,
            'config': self.config.to_dict(),
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_f1 = checkpoint['best_f1']
        logger.info(f"Loaded checkpoint from {path}")


def train_model(model, optimizer, train_dataloader, val_dataset, num_epochs):
    """
    Simplified training function matching your notebook implementation.
    """
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for step, batch in enumerate(tqdm(train_dataloader)):
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad()
            
            running_loss += loss.item()
            
            if step % 50 == 0 and step > 0:
                avg_loss = running_loss / 50
                print(f"Epoch {epoch + 1}, Step {step} | Avg Loss: {avg_loss:.4f}")
                running_loss = 0.0
        
        from .evaluator import evaluate_model
        val_f1 = evaluate_model(model, val_dataset, show_conf_m=False)
        print(f"Epoch {epoch + 1} | Validation F1: {val_f1:.4f}")
    
    return model