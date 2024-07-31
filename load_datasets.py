import os
from datasets import load_dataset

TOKEN = os.environ['HF_TOKEN']

voxpopuli = load_dataset("esb/datasets", "voxpopuli", trust_remote_code=True, token=TOKEN)
cv_13 = load_dataset("mozilla-foundation/common_voice_13_0", "en", trust_remote_code=True, token=TOKEN)

voxpopuli = voxpopuli.remove_columns(['id', 'audio', 'dataset'])
cv_13 = cv_13.remove_columns(['client_id', 'path', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment', 'variant'])
cv_13 = cv_13.rename_column('sentence', 'text')

voxpopuli.save_to_disk('data/voxpopuli')
cv_13.save_to_disk('data/cv_13')