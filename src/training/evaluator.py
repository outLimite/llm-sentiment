"""
Model evaluation and inference utilities.
"""

import torch
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
import logging

from ..utils.postprocessing import SentimentPostprocessor

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Model evaluation class with comprehensive metrics."""
    
    def __init__(self, tokenizer, postprocessor: Optional[SentimentPostprocessor] = None):
        self.tokenizer = tokenizer
        self.postprocessor = postprocessor or SentimentPostprocessor()
        
        self.IDX2NAME = {0: "negative", 1: "neutral", 2: "positive"}
        self.NAME2IDX = {v: k for k, v in self.IDX2NAME.items()}
        self.NAME2IDX[""] = len(self.NAME2IDX)  

    @torch.no_grad()
    def evaluate(
        self, 
        model, 
        dataset, 
        batch_size: int = 100,
        show_conf_m: bool = True,
        max_new_tokens: int = 16
    ) -> float:
        """
        Evaluate model on dataset.
        
        Args:
            model: Model to evaluate
            dataset: Evaluation dataset
            batch_size: Batch size for evaluation
            show_conf_m: Whether to show confusion matrix
            max_new_tokens: Maximum new tokens for generation
            
        Returns:
            Macro F1 score
        """
        model.eval()
        ground_truth = []
        predicted = []
        
        for examples in tqdm(dataset.batch(batch_size), desc="Evaluating"):
            input_ids, attention_mask = self._prepare_inputs(examples)

            texts = self._generate_predictions(
                model, input_ids, attention_mask, max_new_tokens
            )

            batch_truth, batch_pred = self._process_batch(examples, texts)
            ground_truth.extend(batch_truth)
            predicted.extend(batch_pred)
        
        f1 = self._calculate_metrics(ground_truth, predicted, show_conf_m)
        
        return f1

    def _prepare_inputs(self, examples: Dict[str, Any]):
        """Prepare input tensors for generation."""
        from .utils import pad_tensors
        
        input_ids = [
            torch.tensor(ids, dtype=torch.long) 
            for ids in examples["input_ids"]
        ]
        
        input_ids = pad_tensors(
            input_ids, 
            padding_value=self.tokenizer.pad_token_id,
            padding_side="left"
        ).to(model.device)
        
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        return input_ids, attention_mask

    def _generate_predictions(self, model, input_ids, attention_mask, max_new_tokens: int):
        """Generate predictions for batch."""
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        # Extract generated text (after input)
        shrinked_ids = output_ids[:, input_ids.shape[1]:]
        texts = self.tokenizer.batch_decode(
            shrinked_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )
        
        return texts

    def _process_batch(self, examples: Dict[str, Any], texts: List[str]):
        """Process batch of predictions."""
        batch_truth = []
        batch_pred = []
        
        for i in range(len(examples["str_label"])):
            true_label = examples["str_label"][i]
            batch_truth.append(self.NAME2IDX[true_label])
            
            predicted_sentiment = self.postprocessor.extract_sentiment(texts[i])
            batch_pred.append(self.NAME2IDX.get(predicted_sentiment, self.NAME2IDX[""]))
        
        return batch_truth, batch_pred

    def _calculate_metrics(self, ground_truth, predicted, show_conf_m: bool):
        """Calculate evaluation metrics."""
        f1 = f1_score(
            ground_truth, 
            predicted,
            labels=list(self.NAME2IDX.values()),
            average="macro", 
            zero_division=0.0
        )
        
        if show_conf_m:
            self._plot_confusion_matrix(ground_truth, predicted)
        
        logger.info(f"Macro F1 Score: {f1:.4f}")
        return f1

    def _plot_confusion_matrix(self, ground_truth, predicted):
        """Plot confusion matrix."""
        conf_m = confusion_matrix(
            ground_truth, 
            predicted, 
            labels=list(self.NAME2IDX.values())
        )
        
        disp = ConfusionMatrixDisplay(
            conf_m, 
            display_labels=list(self.NAME2IDX.keys())
        )
        
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, cmap='Blues')
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()


@torch.no_grad()
def evaluate_model(
    model, 
    dataset, 
    tokenizer, 
    show_conf_m: bool = True, 
    batch_size: int = 100
) -> float:
    """
    Evaluate model (functional version from your notebook).
    """
    evaluator = ModelEvaluator(tokenizer)
    return evaluator.evaluate(
        model, 
        dataset, 
        batch_size=batch_size, 
        show_conf_m=show_conf_m
    )


def generate_class(model, tokenizer, input_ids):
    """
    Generate classification for single input.
    
    Args:
        model: Model for generation
        tokenizer: Tokenizer
        input_ids: Input token IDs
        
    Returns:
        Generated text
    """
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids, device=model.device).unsqueeze(0)
    
    output_ids = model.generate(input_ids, max_new_tokens=16)
    generated_text = tokenizer.decode(
        output_ids[0][len(input_ids[0]):], 
        skip_special_tokens=True
    )
    
    return generated_text


def print_predictions(model, tokenizer, dataset, num_examples: int = 5):
    """
    Print model predictions for examples.
    
    Args:
        model: Model for prediction
        tokenizer: Tokenizer
        dataset: Dataset with examples
        num_examples: Number of examples to print
    """
    postprocessor = SentimentPostprocessor()
    
    print("=== MODEL PREDICTIONS ===")
    for i in range(min(num_examples, len(dataset))):
        input_ids = torch.tensor([dataset[i]["input_ids"]], device=model.device)
        generated_text = generate_class(model, tokenizer, input_ids)
        predicted_sentiment = postprocessor.extract_sentiment(generated_text)
        
        print(f"Text: {dataset[i]['text']}")
        print(f"True: {dataset[i]['str_label']}")
        print(f"Pred: {predicted_sentiment}")
        print(f"Generated: {generated_text}")
        print("=" * 50)