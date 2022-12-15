from huggingface_hub import login
login("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")

import numpy as np
import pandas as pd
from tqdm import tqdm

# datasets_urls = pd.read_csv("../data/asr_dataset_info.csv")

# dataset_file_names = [link.replace("https://storage.googleapis.com/test_public_bucket/labelled/Telugu_","").replace(".zip","") 
#                       for link in datasets_urls.corpus_cownload_link ]
# print(dataset_file_names)


# DOWNLOAD_URLS = datasets_urls[datasets_urls.source.isin(['Zee_Telugu','idream_telugu_movies',
#                                          'telangana_sahitya_akademi',
#                                          'dd_saptagiri',
#                                         'chai_bisket_stories'])].corpus_cownload_link.unique().tolist()
# print(DOWNLOAD_URLS[:2])


# dataset_file_names = [link.replace("https://storage.googleapis.com/test_public_bucket/labelled/Telugu_","").replace(".zip","") 
#                       for link in DOWNLOAD_URLS]
# print("##### Dataset file names #####")
# print(dataset_file_names)

import os
from IPython import display
import datasets
from datasets.info import DatasetInfosDict
from datasets import load_dataset,concatenate_datasets, DatasetDict, Dataset, Audio

# def get_dataset(file_name):
#     ds_base_path = os.getcwd()+"/training_data/"+file_name
#     print(f'ds_base_path --> {ds_base_path}')
#     ds = load_dataset('json', data_files=ds_base_path+"/data.json")
#     ds = ds.map(lambda x: {'audioFilename':ds_base_path+"/"+x["audioFilename"]})
#     return ds

# ds = get_dataset(dataset_file_names[0])['train']
# print(ds)

# print(ds)
# for fn in tqdm(dataset_file_names[1:]):
#     temp_ds = get_dataset(fn)['train']
#     ds = concatenate_datasets([ds,temp_ds])
#     print(ds)


def get_dataset_from_json(file_name):
    ds_base_path = os.getcwd()+"/training_data/"+file_name
    ds = load_dataset('json', data_files=ds_base_path+"/data.json")
    ds = ds.map(lambda x: {'audioFilename':ds_base_path+"/"+x["audioFilename"]})
    ds = ds.remove_columns(['collectionSource', 'snr', 'duration', 'gender'])
    return ds['train']

def get_dataset_from_csv(file_name):
    ds_base_path = os.getcwd()+"/training_data/"+file_name
    ds = load_dataset('csv', data_files=ds_base_path+"/metadata.csv")
    ds = ds.map(lambda x: {'file_name':ds_base_path+"/"+x["file_name"]})
    ds = ds.rename_column("file_name", "audioFilename")
    ds = ds.rename_column("transcription", "text")
    return ds['train']

csv_files = ["noisy_test_unknown","noisy_test_known","clean_test_known","clean_test_unknown","clean_valid","te_in_female","te_in_male"]
ds = get_dataset_from_json("Telangana_Sahitya_Akademi_16-08-2021_14-40/")
print(ds)
for fn in csv_files:
    temp_ds = get_dataset_from_csv(fn)
    ds = concatenate_datasets([ds,temp_ds])
    print(ds)

ds = ds.cast_column("audioFilename",Audio(sampling_rate=16000))
ds = ds.rename_column("audioFilename", "audio")

# print({f"Training on {(np.sum(ds['duration'])/60)/60} hours of data"})


train_test_dataset = ds.train_test_split(test_size=0.15)
# train_test_dataset = train_test_dataset.remove_columns(['collectionSource', 'snr', 'duration', 'gender'])
print(train_test_dataset)

print("####### Loaded all the datasets, ready for tuning#########")

from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Telugu", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Telugu", task="transcribe")


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]
    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["text"]).input_ids
    # batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

print("####### Preparing dataset started ########")
train_test_dataset = train_test_dataset.map(prepare_dataset, remove_columns=train_test_dataset.column_names["train"], num_proc=20)
print("####### Preparing dataset completed ########")


print("####### Define a Data Collator ########")
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

torch.cuda.empty_cache()

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

print("######### Evaluation Metrics #########")

import evaluate

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

print("####### Load a Pre-Trained Checkpoint ########")

from transformers import WhisperForConditionalGeneration
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")


model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

print("####### Define the Training Configuration #########")


from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-te-14k",  # change to a repo name of your choice
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=15000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)


from transformers import Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_test_dataset["train"],
    eval_dataset=train_test_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

processor.save_pretrained(training_args.output_dir)


trainer.train()

print("######## Completed Training #########")

kwargs = {
    "dataset_tags": "Chai_Bisket_Stories_16-08-2021_14-17",
    "dataset": "Chai_Bisket_Stories_16-08-2021_14-17",
    "dataset_args": "config: te, split: test",
    "language": "te",
    "model_name": "Whisper Small Telugu - Naga Budigam",
    "finetuned_from": "openai/whisper-small",
    "tasks": "automatic-speech-recognition",
    "tags": "hf-asr-leaderboard",
}
print("######## Pushing to HF Hub #########")
trainer.push_to_hub(**kwargs)