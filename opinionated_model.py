import os
import torch
import wandb

from trl import SFTTrainer

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig

max_seq_length = 128


class OpinionatedModel:

    def __init__(self, token: str = "", dataset=None, wandb_project: str = ""):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.token = token

        self.lora_config = LoraConfig(
            r=8,
            target_modules=[
                "q_proj",
                "o_proj",
                "k_proj",
                "v_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
        )

        model_id = "google/gemma-2b"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=self.token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map={"": 0},
            token=self.token,
        )

        wandb.init(project=wandb_project)

        self.dataset = dataset
        self.trainer = None

    def prepare_dataset(self, dataset):
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"], truncation=True, max_length=max_seq_length
            )

        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )
        return tokenized_dataset.filter(lambda x: len(x["input_ids"]) > 8)

    def save_tokenized_dataset(self, save_directory, folder_name):
        folder_path = os.path.join(save_directory, folder_name)

        if os.path.exists(folder_path):
            print(f"{folder_name} dataset already exists!")
        else:
            # Save the dataset to the specified directory
            self.prepare_dataset(self.dataset).save_to_disk(folder_path)
            print(f"Dataset saved to '{folder_path}'")

    def get_dataset(
        self, dataset, split: str, smaller_version: bool = True, num_samples: int = 1000
    ):
        return (
            dataset[split].shuffle(seed=42).select(range(num_samples))
            if smaller_version
            else dataset[split]
        )

    def _create_trainer(
        self,
        on_small_dataset: bool = True,
        num_samples: int = 1000,
        num_epochs: int = 2,
        tokenize_dataset: bool = True,
    ) -> None:
        if tokenize_dataset:
            train_dataset = self.prepare_dataset(
                self.get_dataset(self.dataset, "train", on_small_dataset, num_samples)
            )
            eval_dataset = self.prepare_dataset(
                self.get_dataset(self.dataset, "validation", True, 16000)
            )
        else:
            train_dataset = self.get_dataset(
                self.dataset, "train", on_small_dataset, num_samples
            )
            eval_dataset = self.get_dataset(self.dataset, "validation", True, 16000)

        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            max_seq_length=max_seq_length,
            args=TrainingArguments(
                per_device_train_batch_size=4,
                gradient_accumulation_steps=2,
                warmup_steps=2,
                eval_strategy="steps",
                eval_steps=7500,
                save_steps=7500,
                num_train_epochs=num_epochs,
                learning_rate=2e-5,
                fp16=True,
                logging_steps=1,
                output_dir="output_opinionated",
                optim="paged_adamw_8bit",
                weight_decay=0.02,
                eval_on_start=True,
            ),
            peft_config=self.lora_config,
            dataset_text_field="text",
        )

    def train(
        self,
        on_small_dataset: bool = True,
        num_samples: int = 1000,
        num_epochs: int = 2,
        tokenize_dataset: bool = True,
    ) -> None:
        self._create_trainer(
            on_small_dataset, num_samples, num_epochs, tokenize_dataset
        )
        self.trainer.train()
        wandb.finish()
