import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.data.dataset import load_tweet_dataset
from src.data.preprocessing import TokenizerWrapper, preprocess_dataset
from src.training.configs import get_lora_config
from src.models.base import setup_device, apply_peft_to_model, freeze_layers
from src.training.trainer import PEFTTrainerWithNeptune
from src.models.lora import LinearWithLoRA

from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader
import neptune


run = neptune.init_run(
    project="trapcodd1e/llm-sentiment", 
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhZWZkYzBkYS05ZWFhLTQwMjgtYTc4Yi1lYzBiYjgyYTVkODUifQ==",
    tags=["peft", "lora", "tweet-classification"],
)

MODEL_NAME = "OuteAI/Lite-Oute-1-300M-Instruct"

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer_wrapper = TokenizerWrapper(MODEL_NAME)

device = setup_device()
model.to(device)

dataset = load_tweet_dataset(cache_dir="~/.cache/huggingface/datasets")

processed_dataset = preprocess_dataset(dataset, tokenizer_wrapper.tokenizer)

collate_fn = tokenizer_wrapper.get_collate_fn(
    pad_token_id=tokenizer_wrapper.tokenizer.pad_token_id
)

config = get_lora_config()

apply_peft_to_model(model, LinearWithLoRA, config.lora_rank, config.lora_alpha, config.target_modules)

model = freeze_layers(model, ["lora"])

train_dataloader = DataLoader(
    processed_dataset["train"],
    batch_size=config.batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)

trainer = PEFTTrainerWithNeptune(
    "OuteAI/Lite-Oute-1-300M-Instruct",
    model, 
    train_dataloader, 
    processed_dataset["validation"], 
    tokenizer_wrapper.tokenizer,
    config,
)

trainer.train()

trainer.save_checkpoint("checkpoint/OuteAI/Lite-Oute-1-300M-Instruct")
