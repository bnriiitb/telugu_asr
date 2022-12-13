from transformers import AutoProcessor, AutoModelForCTC
processor = AutoProcessor.from_pretrained("Harveenchadha/vakyansh-wav2vec2-telugu-tem-100")
model = AutoModelForCTC.from_pretrained("Harveenchadha/vakyansh-wav2vec2-telugu-tem-100")

import numpy as np
import pandas as pd
import os
from IPython import display
import datasets
from datasets.info import DatasetInfosDict
from datasets import load_dataset,concatenate_datasets, DatasetDict, Dataset, Audio
import torch

fleurs_dataset = load_dataset("google/fleurs", "te_in", split="test",num_proc=32)
fleurs_dataset = fleurs_dataset.rename_column("transcription", "text")
fleurs_dataset = fleurs_dataset.cast_column("audio",Audio(sampling_rate=16000))
fleurs_dataset = fleurs_dataset.remove_columns(['id', 'num_samples', 'raw_transcription', 'gender', 'lang_id', 'language', 'lang_group_id'])
print(fleurs_dataset)

sampling_rate = fleurs_dataset.features["audio"].sampling_rate

def transcribe(audio_array):
    inputs = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    # transcribe speech
    transcription = processor.batch_decode(predicted_ids)
    transcription = transcription[0].replace("<s>","")
    return transcription

from tqdm import tqdm
vakyansh_texts = []
for item in tqdm(fleurs_dataset['audio']):
    vakyansh_texts.append(transcribe(item['array']))
print(vakyansh_texts)


import evaluate
metric = evaluate.load("wer")
vakyansh_wer = 100 * metric.compute(predictions=vakyansh_texts, references=fleurs_dataset['text'])
print(vakyansh_wer)