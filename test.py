import os
import torch
import math

from dotenv import load_dotenv

from datasets import load_from_disk
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

load_dotenv()

TOKEN = os.getenv("HF_TOKEN")

OPINIONATED_DATA_PATH = "data/opinionated"
INFORMATIVE_DATA_PATH = "data/common_voice_13_0"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

informative_model = AutoModelForCausalLM.from_pretrained("merged/informative_merged", quantization_config=bnb_config, device_map={"": 0})
opinionated_model = AutoModelForCausalLM.from_pretrained("merged/opinionated_merged", quantization_config=bnb_config, device_map={"": 0})

tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b')


def get_sentence_probability(sentence):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenized_sentence = tokenizer.encode(sentence, return_tensors='pt')
    sentence_prob = 0

    inputs = tokenized_sentence.to(device)
    with torch.no_grad():
        outputs_informative = informative_model.model(inputs)[0]
        outputs_opinionated = opinionated_model.model(inputs)[0]
    for i, token in enumerate(inputs[0][1:]):
        p2 = torch.nn.functional.softmax(outputs_informative[0, i, :], dim=-1)[token].item() * 100
        p3 = torch.nn.functional.softmax(outputs_opinionated[0, i, :], dim=-1)[token].item() * 100
        log_diff = math.log(p3/p2)
        sentence_prob += log_diff
    return sentence_prob


opinionated_set = load_from_disk(OPINIONATED_DATA_PATH)
informative_set = load_from_disk(INFORMATIVE_DATA_PATH)

correct = 0
for sentence in informative_set['test'][:]['text']:
    prob = get_sentence_probability(sentence)
    if prob < 0:
        correct += 1
    print(f'{sentence}: {prob}')
for sentence in opinionated_set['test'][:]['text']:
    prob = get_sentence_probability(sentence)
    if prob > 0:
        correct += 1
    print(f'{sentence}: {prob}')
score = (correct / (len(informative_set['test']) + len(opinionated_set['test']))) * 100
print(f'Final score: {score}')
