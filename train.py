import transformers
import torch.nn.functional as F
from datasets import load_dataset
from safetensors.torch import save_model
from model import Simple1Model
from config import ModelConfig, TrainingConfig
from utils import preprocess_data, SafeSaveCallback
from huggingface_hub import login
from dotenv import load_dotenv
import os
import torch
import trl
from transformers import TrainingArguments  # Ensure this is imported

load_dotenv()
login(os.getenv('HF_KEY'))

def train():
	model_cfg = ModelConfig()
	train_cfg = TrainingConfig()

	model = transformers.AutoModelForCausalLM.from_pretrained(model_cfg.model_name)
	dataset = load_dataset(train_cfg.dataset_name)

	if isinstance(dataset, dict):
		dataset = dataset['train']

	tokenizer = transformers.AutoTokenizer.from_pretrained(model_cfg.model_name, use_fast=True)

	instruction_template = "<|im_start|>user"
	response_template = "<|im_start|>assistant\n"
	tokenizer.pad_token = "<|fim_pad|>"

	collator = trl.DataCollatorForCompletionOnlyLM(
		instruction_template=instruction_template,
		response_template=response_template,
		tokenizer=tokenizer,
		mlm=False
	)

	training_args = TrainingArguments(
		output_dir=train_cfg.output_dir,
		per_device_train_batch_size=train_cfg.batch_size,
		per_device_eval_batch_size=train_cfg.batch_size,
		gradient_accumulation_steps=train_cfg.gradient_accumulation,
		learning_rate=train_cfg.learning_rate,
		num_train_epochs=train_cfg.epochs,
		bf16=True,
		logging_steps=100,
		remove_unused_columns=False,
		save_strategy="no",
		save_only_model=True,
		dataset_text_field="text"
	)

	trainer = trl.SFTTrainer(
		model=model,
		train_dataset=dataset['train'],
		eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'],
		args=training_args,
		data_collator=collator
	)

	trainer.train()
	trainer.save_model(train_cfg.output_dir)

if __name__ == "__main__":
    train()
