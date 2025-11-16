"""
Data loading and preparation for tweet sentiment analisys
"""

from datasets import load_dataset
from typing import Dict, Optional
import logging 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="logging/app.log",
    filemode="w",
    )
logger = logging.getLogger(__name__)

IDX2NAME = {0: "negative", 1: "neutral", 2: "positive"}
NAME2IDX = {v : k for k, v in IDX2NAME.items()}

def load_tweet_dataset(
    dataset_name: str = "cardiffnlp/tweet_eval",
    task: str = "sentiment",
    cache_dir: Optional[str] = None
    ) -> Dict[str, any]:
    """
    Load and prepare the tweet sentiment dataset. 

    Args:
        dataset_name: Name of the dataset on HuggingFace Hub
        task: Specific task/subset of the dataset 
        cache_dir: Directory to cache the dataset 

    Returns:    
        Dictionary with train/validation/test splits
    """
    try:
        logger.info(f"Loading dataset: {dataset_name} task: {task}")
        dataset = load_dataset(dataset_name, task, cache_dir=cache_dir)

        # Add string labels for readability
        def add_str_label(example):
            example["str_label"] = IDX2NAME[example["label"]]
            return example
        
        for split, data in dataset.items():
            dataset[split] = data.map(add_str_label)

        logger.info(f"Dataset loaded succesfully. Splits: {list(dataset.keys())}")
        logger.info(f"Train set size {len(dataset["train"])}")
        logger.info(f"Validation set size {len(dataset["validation"])}")
        logger.info(f"Test set size {len(dataset["test"])}")

        return dataset

    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def get_dataset_stats(dataset: Dict[str, any]) -> Dict[str, Dict[str, int]]:
    """
    Calculate statistics for the dataset.

    Args:
        dataset: Loaded dataset with splits
    
    Returns 
        Dictionary with statistics per splits
    """
    stats = {}

    for split, data in dataset.items():
        label_counts = {}
        for example in data:
            label = example["str_label"]
            label_counts[label] = label_counts.get(label, 0) + 1

        stats[split] = {
            "total_examples": len(data),
            "label_distribution": label_counts
        }

    return stats

def print_dataset_samples(dataset: Dict[str, int], num_samples: int = 3):
    """
    Print sample examples from the dataset

    Args:
        dataset: Loaded dataset 
        num_samples: Number of samples to print per split 
    """
    for split, data in dataset.items():
        print(f"\n==={split.upper()} SAMPLES===")
        for i in range(min(len(data), num_samples)):
            example = dataset[split][i]
            print(f"Text: {example["text"]}")
            print(f"Label: {example["str_label"]}")
            print("-"*50)


if __name__  == "__main__":
    dataset = load_tweet_dataset(cache_dir="~/.cache/huggingface/datasets")
    stats = get_dataset_stats(dataset)
    print_dataset_samples(dataset, 1)
    
    print("\n===Dataset STATS===")
    for split, split_stats in stats.items():
        print(f"{split}: {split_stats}")