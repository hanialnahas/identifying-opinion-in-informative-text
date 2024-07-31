# identifying-opinion-in-informative-text

## Overview

We have fine-tuned two GEMMA models to identify words that pushes opinion in text as well as words that gives information. we then combine the probability score from the two models to output a score for each word as well as the whole sentence.

## Datasets

- For training the informative model we are using the *common_voice_13_0 dataset*.
- For the opinionated model we are using a mix of the *voxpopuli* european parliment and the *amazon polarity* amazon reviews datasets.

## Usage
1. create a ```.env``` file with two environment variables:
    - **HF_TOKEN**: Hugging Face Token
    - **WANDB_PROJECT**: WAND Porject Name 
2. run ```pip install -r requirements.txt```
3. run ```python load_datasets.py``` to load and clean the two datasets used to train the models. This script takes a lot of time and disk space (400 GB). You can also download and unzip the datasets in the *data* folder from the links below:
    - [**Opinionated**](https://drive.google.com/file/d/1xXU8bQM1Gyuo1sS2x0vlDbDY_2KmLpey/view?usp=sharing)
    - [**Common_Voice**](https://drive.google.com/file/d/1Nl10f0YRvhbaBmu-JuATxNa9Ji5EL_MZ/view?usp=sharing)
4. run ```python train.py```:
   - By default, the pretrained model lora weights in the *pretrained* folder would be merged with the base model.
   - Set ```USE_PRETRAINED = False``` to train the two models from scratch and merge the lora weights with the base model. 
5. run ```python test.py``` to test the models on the test datasets and output a final score.