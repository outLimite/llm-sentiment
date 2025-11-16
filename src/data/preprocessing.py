"""
Data preprocessing and tokenization for PEFT training. 
"""
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from functools import partial
from typing import Optional, Dict, List, Any
from pathlib import Path
import logging 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="logging/app.log",
    filemode="w",
    )
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Handles data preprocessing and promt formatting 
    """

    def __init__(self, system_prompt: Optional[str] = None, user_prompt: Optional[str] = None):

        base_dir = Path(__file__).resolve().parents[2]  

        prompts_dir = base_dir / "prompts"
        if not system_prompt: 
            with open(prompts_dir / "system_prompt.txt", mode="r", encoding="utf-8") as f:
                self.system_prompt = f.read()
        else:
            self.system_prompt = system_prompt

        if not user_prompt:
            with open(prompts_dir / "user_prompt.txt", mode="r", encoding="utf-8") as f:
                self.user_prompt = f.read()
        else:
            self.user_prompt = user_prompt

    def process_example(
        self, 
        example: Dict[str, Any], 
        tokenizer: AutoTokenizer
        ) -> Dict[str, Any]:
        """Processes a single example by constructing a chat-based prompt and tokenizing it.

        Process:
            1. Constructs a conversation comprising three roles:
                - "system": Provides instructions for classifying the sentiment.
                - "user": Presents the input message.
                - "assistant": Contains the expected sentiment answer.
            2. Applies the chat template to generate the full prompt.

        Parameters:
            example (dict): A dictionary with keys:
                - 'text': The message to be classified.
                - 'str_label': The expected sentiment label.
            system_prompt (str): System prompt to use

        Returns:
            dict: Extended dictionary containing conversational prompts:
                - 'prompt': Input task
                - 'full_prompt': Full conversation including task label
        """
        user_content = self.user_prompt.format(text=example["text"])

        messages = [
            {"role": "system", "content": self.system_prompt,},
            {"role": "user", "content": user_content}
        ]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        full_messages = messages + [
            {"role": "assistant", "content": example["str_label"]}
        ]

        full_prompt = tokenizer.apply_chat_template(full_messages, tokenize=False)

        return {
            "prompt": prompt,
            "full_prompt": full_prompt,
        }
    
    def batch_process(
        self, 
        examples: Dict[str, List], 
        tokenizer,
        include_assistant: bool = True
        ) -> Dict[str, List]:
        """
        Process a batch of examples.
        
        Args:
            examples: Batch of examples
            tokenizer: Tokenizer for processing
            include_assistant: Whether to include assistant responses
            
        Returns:
            Batch processed examples
        """
        processed_batch = {
            "text": [],
            "label": [],
            "str_label": [],
            "prompt": [],
            "messages": []
        }
        
        if include_assistant:
            processed_batch["full_prompt"] = []
            processed_batch["full_messages"] = []

        for i in range(len(examples["text"])):
            example = {
                "text": examples["text"][i],
                "label": examples["label"][i] if "label" in examples else None,
                "str_label": examples["str_label"][i] if "str_label" in examples else None,
            }
            
            processed = self.process_example(example, tokenizer)

            processed_batch["prompt"].append(processed["prompt"])

            if include_assistant:
                processed_batch["full_prompt"].append(processed["full_prompt"])

        return processed_batch

class TokenizerWrapper:
    """
    Wrapper for tokenizer with additional utilities
    """

    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.setup_tokenizer()

    def setup_tokenizer(self):
        """Setup tokenizer with appropriate settings"""
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def tokenization(
        self, 
        examples: Dict[str, List],
        max_length: int = 256) -> Dict[str, List]:
        """
        Tokenize a batch of examples.
        
        Args:
            examples: Batch of examples with prompts
            max_length: Maximum sequence length
        
        Returns:
            Tokenized examples
        """
        prompt_enc = self.tokenizer(
            examples["prompt"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        full_prompt_enc = self.tokenizer(
            examples["full_prompt"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        return {
            "input_ids": prompt_enc["input_ids"],
            "attention_mask": prompt_enc["attention_mask"],
            "full_input_ids": full_prompt_enc["input_ids"],
            "full_attention_mask": full_prompt_enc["attention_mask"],
        }

    def get_collate_fn(self, pad_token_id: int = 0):
        """
        Get collate function for DataLoader 
        """
        def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
            """
            Collate and pad a batch of tokenized examples.
            """
            full_input_ids = [torch.tensor(ex["full_input_ids"], dtype=torch.long) 
                            for ex in batch if "full_input_ids" in ex]
            
            if not full_input_ids:
                raise ValueError("No full_input_ids found in batch")

            max_length = max(t.size(0) for t in full_input_ids)
            padded_input_ids = []
            attention_masks = []
            
            for t in full_input_ids:
                pad_len = max_length - t.size(0)
                if pad_len > 0:
                    pad_tensor = torch.full((pad_len,), pad_token_id, 
                                        dtype=t.dtype, device=t.device)
                    # Сохраняем оригинальный тензор до конкатенации
                    original_t = t
                    t = torch.cat([original_t, pad_tensor], dim=0)
                    # Создаем маску: 1 для реальных токенов, 0 для паддинга
                    mask = torch.cat([
                        torch.ones_like(original_t), 
                        torch.zeros_like(pad_tensor)
                    ], dim=0)
                else:
                    # Если паддинг не нужен
                    mask = torch.ones_like(t)
                
                padded_input_ids.append(t)
                attention_masks.append(mask)
            
            batch_out = {
                "input_ids": torch.stack(padded_input_ids),
                "attention_mask": torch.stack(attention_masks),
                "labels": torch.stack(padded_input_ids), 
            }
            
            return batch_out
    
        return collate_fn
    
def preprocess_dataset(
    dataset: Dict[str, Dataset],
    tokenizer: AutoTokenizer,
    system_prompt: Optional[str] = None,
    max_length: int = 256
) -> Dict[str, Dataset]:
    """
    Preprocess entire dataset for PEFT training.
    
    Args:
        dataset: Raw dataset splits
        tokenizer: Tokenizer for encoding
        system_prompt: System prompt for chat format
        max_length: Maximum sequence length
    
    Returns:
        Preprocessed dataset splits
    """
    processor = DataProcessor(system_prompt)
    tokenizer_wrapper = TokenizerWrapper(tokenizer.name_or_path)
    tokenizer_wrapper.tokenizer = tokenizer
    
    processed_datasets = {}
    
    for split_name, split_data in dataset.items():
        logger.info(f"Preprocessing {split_name} split...")
        
        split_data = split_data.map(
            processor.process_example,
            fn_kwargs={"tokenizer": tokenizer},
            desc=f"Formatting {split_name}"
        )
        
        split_data = split_data.map(
            lambda examples: tokenizer_wrapper.tokenization(examples, max_length=max_length),
            batched=True,
            desc=f"Tokenizing {split_name}"
        )
        
        processed_datasets[split_name] = split_data
    
    return processed_datasets


if __name__ == "__main__":

    processor = DataProcessor()
    
    tokenizer = AutoTokenizer.from_pretrained("OuteAI/Lite-Oute-1-300M-Instruct")
    
    test_example = {
        "text": "QT @user In the original draft of the 7th book, Remus Lupin survived the Battle of Hogwarts. #HappyBirthdayRemusLupin",
        "label": 2,
        "str_label": "positive"
    }
    
    result = processor.process_example(test_example, tokenizer=tokenizer)
    
    print("=== DATA PROCESSOR TEST ===")
    print("System prompt:", processor.system_prompt)
    print("User prompt template:", processor.user_prompt)
    print("\nOriginal text:", test_example["text"])
    print("Label:", test_example["str_label"])
    print("\nGenerated prompt (first 150 chars):")
    print(result["prompt"][:150] + "...")
    print("\nFull prompt (first 150 chars):")
    print(result["full_prompt"][:150] + "...")
    
    print("\n=== BATCH PROCESSING TEST ===")
    batch_examples = {
        "text": [
            "I love this movie! It's amazing.",
            "This is terrible, worst experience ever.",
            "The weather is okay today."
        ],
        "label": [2, 0, 1],
        "str_label": ["positive", "negative", "neutral"]
    }
    
    batch_result = processor.batch_process(batch_examples, tokenizer=tokenizer)
    print(f"Processed {len(batch_result['text'])} examples in batch")
    print("First batch prompt length:", len(batch_result["prompt"][0]))
    
    print("\nDataProcessor test completed successfully!")