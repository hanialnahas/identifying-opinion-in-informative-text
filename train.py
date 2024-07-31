import os
import torch
import peft
from datasets import load_from_disk
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from dotenv import load_dotenv

from informative_model import InformativeModel
from opinionated_model import OpinionatedModel

load_dotenv()

TOKEN = os.getenv("HF_TOKEN")
print(TOKEN)
WANDB_PROJECT = os.getenv("WANDB_PROJECT")

OPINIONATED_DATA_PATH = "data/opinionated"
INFORMATIVE_DATA_PATH = "data/common_voice_13_0"

OPINIONATED_SAVE_PATH = "trained/opinionated_model"
INFORMATIVE_SAVE_PATH = "trained/informative_model"

opinionated_set = load_from_disk(OPINIONATED_DATA_PATH)
opinionated_model = OpinionatedModel(TOKEN, opinionated_set, WANDB_PROJECT)
opinionated_model.train(on_small_dataset=False)
opinionated_model.trainer.save_model(OPINIONATED_SAVE_PATH)

informative_set = load_from_disk(INFORMATIVE_DATA_PATH)
informative_model = InformativeModel(TOKEN, informative_set, WANDB_PROJECT)
informative_model.train(on_small_dataset=False)
informative_model.trainer.save_model(INFORMATIVE_SAVE_PATH)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
base_model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", quantization_config=bnb_config, device_map={"": 0}, token=TOKEN)

opinionated_merged = peft.PeftModel.from_pretrained(base_model, OPINIONATED_SAVE_PATH)
opinionated_merged.merge_and_unload().save_pretrained("merged/opinionated_merged")

informative_merged = peft.PeftModel.from_pretrained(base_model, INFORMATIVE_SAVE_PATH)
informative_merged.merge_and_unload().save_pretrained("merged/informative_merged")
