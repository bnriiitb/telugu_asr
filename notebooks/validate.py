# import numpy as np
# import pandas as pd
# import os
# from IPython import display
# import datasets
# from datasets.info import DatasetInfosDict
# from datasets import load_dataset,concatenate_datasets, DatasetDict, Dataset, Audio
# from datasets import load_dataset
# from transformers import WhisperForConditionalGeneration, WhisperProcessor
# import soundfile as sf
# import torch
# import time
# from tqdm import tqdm
# from transformers import pipeline
# import whisper
# import evaluate


# metric = evaluate.load("wer")


# print("##### Loading the dataset #####")
# fleurs_dataset = load_dataset("google/fleurs", "te_in", split="test",num_proc=32)
# fleurs_dataset = fleurs_dataset.rename_column("transcription", "text")
# fleurs_dataset = fleurs_dataset.cast_column("audio",Audio(sampling_rate=16000))
# fleurs_dataset = fleurs_dataset.remove_columns(['id', 'num_samples', 'raw_transcription', 'gender', 'lang_id', 'language', 'lang_group_id'])
# print(fleurs_dataset)

# print("###### Loading finetuned model started #####")
# pipe = pipeline(model="bnriiitb/whisper-small-te",language='Telugu')
# print("###### Loading finetuned model completed #####")

# print("###### Loading whispher small model started #####")
# whisper_small_model = whisper.load_model("small",device='cuda')
# print("###### Loading whispher small model completed #####")


# print("###### Transcribing Text using finetunred Model started")
# finetuned_texts = []
# for item in tqdm(fleurs_dataset['audio']):
#     finetuned_texts.append(pipe(item['array'],language='Telugu')['text'])
# print("###### Transcribing Text using finetunred Model completed")
# print(finetuned_texts)

# print("##### Computing WER on finetuned model started #####")
# finetuned_wer = 100 * metric.compute(predictions=finetuned_texts, references=fleurs_dataset['text'])
# print("##### Computing WER on finetuned model completed #####")
# print(f'Finetuned Model WER --> {finetuned_wer}')

# whisper_langs = []
# whisper_texts = []
# print("###### Transcribing Text using whisper small Model started")
# for item in tqdm(fleurs_dataset['path']):
#     pred = whisper_small_model.transcribe(item,language='Telugu')
#     whisper_texts.append(pred['text'])
#     whisper_langs.append(pred['language'])
# print("###### Transcribing Text using whisper small Model completed")
# print(whisper_texts)


# print("##### Compute WER started #####")
# finetuned_wer = 100 * metric.compute(predictions=finetuned_texts, references=fleurs_dataset['text'])
# whisper_wer = 100 * metric.compute(predictions=whisper_texts, references=fleurs_dataset['text'])
# print("##### Compute WER Comlpeted #####")
# print(f"whisper_wer --> {whisper_wer}, finetuned_wer --> {finetuned_wer}")