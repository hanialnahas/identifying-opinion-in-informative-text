import os
from dotenv import load_dotenv
from datasets import load_dataset, DatasetDict, concatenate_datasets

load_dotenv()

TOKEN = os.getenv('HF_TOKEN')

voxpopuli = load_dataset("esb/datasets", "voxpopuli", trust_remote_code=True, token=TOKEN)
cv_13 = load_dataset("mozilla-foundation/common_voice_13_0", "en", trust_remote_code=True, token=TOKEN)
amazon_polarity = load_dataset("fancyzhx/amazon_polarity", token=TOKEN)

cv_13 = cv_13.remove_columns(['client_id', 'path', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment', 'variant'])
cv_13 = cv_13.rename_column('sentence', 'text')
cv_13.save_to_disk('data/common_voice_13_0')

voxpopuli = voxpopuli.remove_columns(['id', 'audio', 'dataset'])

amazon_polarity.remove_columns(['label', 'title'])
amazon_polarity.rename_column('content', 'text')

combined_dataset = DatasetDict()

for split, num_samples in [('train', 1000000 - len(voxpopuli["train"])), ('validation', 16000 - len(voxpopuli["validation"]))]:
    amazon_polarity[split] = amazon_polarity[split].shuffle(seed=42).select(range(num_samples))
    combined_dataset[split] = concatenate_datasets([amazon_polarity[split], voxpopuli[split]])

combined_dataset["test"] = amazon_polarity["test"].shuffle(seed=42).select(range(16000))

combined_dataset.save_to_disk('data/opinionated')
