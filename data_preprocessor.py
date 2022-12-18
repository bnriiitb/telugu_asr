import numpy as np
import pandas as pd
import os
from pydub import AudioSegment

BASE_PATH = "/raid/cs20mds14030/telugu_asr/data"
print(BASE_PATH)

INDIC_SUPREB_DATASETS = ["indic_superb/clean_train","indic_superb/clean_valid",
                         "indic_superb/clean_test_known","indic_superb/clean_test_unknown"]

OPENSLR_DATASETS = ["open_slr/te_in_female","open_slr/te_in_male"]

ULCA_DATASETS = ["ulca/BBC_News_Telugu_17-08-2021_00-57",
                 "ulca/Chai_Bisket_Stories_16-08-2021_14-17",
                 "ulca/Telangana_Sahitya_Akademi_16-08-2021_14-40"]

MUCS_DATASETS = ["mucs/te-in-Test/Audios","mucs/te-in-Train/Audios"]



def get_audio_len(file_name):
    sound = AudioSegment.from_file(file_name)
    return round((sound.duration_seconds % 60),3)

def create_metadata_for_indicsuperb_datasets():

    for dataset in INDIC_SUPREB_DATASETS:    
        df = pd.read_csv(f'{BASE_PATH}/{dataset}/transcription_n2w.txt', header=None,sep="\t",names=['file_name','transcription'])
        # replace .m4a with wav
        df.file_name = df.file_name.apply(lambda x: x.replace(".m4a",".wav"))
        
        # add audio duration column
        df.loc[:,"duration"] = df.file_name.apply(lambda file_name: get_audio_len(f"{BASE_PATH}/{dataset}/{file_name}"))
        
        df.to_csv(f'{BASE_PATH}/{dataset}/metadata.csv',index=False)
        # show sample audio
        sample_audio_path = f"{BASE_PATH}/{dataset}/{df.head(1).values[0][0]}"
        sample_audio_transcript = df.head(1).values[0][1]
        print(f"sample_audio_path --> {sample_audio_path}")
        print(f"sample_audio_transcript --> {sample_audio_transcript}")
        print(f'created metadata.csv for {dataset}; contains --> {df.shape[0]}, {round(df.duration.sum(),2)} sec')
        del df
        print("\n")

def create_metadata_for_openslr_datasets():

    for dataset in OPENSLR_DATASETS:    
        df = pd.read_csv(f'{BASE_PATH}/{dataset}/line_index.tsv', header=None,sep="\t",names=['file_name','transcription'])
        # add .wav suffix to all the files
        df.file_name = df.file_name+".wav"
        
        # add audio duration column
        df.loc[:,"duration"] = df.file_name.apply(lambda file_name: get_audio_len(f"{BASE_PATH}/{dataset}/{file_name}"))
        
        df.to_csv(f'{BASE_PATH}/{dataset}/metadata.csv',index=False)
        # show sample audio
        sample_audio_path = f"{BASE_PATH}/{dataset}/{df.head(1).values[0][0]}"
        sample_audio_transcript = df.head(1).values[0][1]
        print(f"sample_audio_path --> {sample_audio_path}")
        print(f"sample_audio_transcript --> {sample_audio_transcript}")
        # display.Audio(sample_audio_path,rate=16000)
        print(f'created metadata.csv for {dataset}; contains --> {df.shape[0]}, {round(df.duration.sum(),2)} sec')
        del df
        print("\n")

def create_metadata_for_ulca_datasets():

    for dataset in ULCA_DATASETS:    
        df = pd.read_json(f'{BASE_PATH}/{dataset}/data.json')
        df.rename(columns={'text':'transcription','audioFilename':'file_name'},inplace=True)
        df = df[['file_name', 'duration','transcription']]        
        df.to_csv(f'{BASE_PATH}/{dataset}/metadata.csv',index=False)
        # show sample audio
        sample_audio_path = f"{BASE_PATH}/{dataset}/{df.head(1).values[0][0]}"
        sample_audio_transcript = df.head(1).values[0][2]
        print(f"sample_audio_path --> {sample_audio_path}")
        print(f"sample_audio_transcript --> {sample_audio_transcript}")
        print(f'created metadata.csv for {dataset}; contains --> {df.shape[0]}, {round(df.duration.sum(),2)} sec')
        del df
        print("\n")

def create_metadata_for_mucs_datasets():

    for dataset in MUCS_DATASETS:
        df = pd.read_csv(f'{BASE_PATH}/{dataset}/transcription.txt',dtype=str,header=None,sep="\t",names=['file_name','transcription'])
        df.file_name = df.file_name.apply(lambda x: str(x)+".wav")
        
        # add audio duration column
        df.loc[:,"duration"] = df.file_name.apply(lambda file_name: get_audio_len(f"{BASE_PATH}/{dataset}/{file_name}"))
        
        df.to_csv(f'{BASE_PATH}/{dataset}/metadata.csv',index=False)
        # show sample audio
        sample_audio_path = f"{BASE_PATH}/{dataset}/{df.head(1).values[0][0]}"
        sample_audio_transcript = df.head(1).values[0][1]
        print(f"sample_audio_path --> {sample_audio_path}")
        print(f"sample_audio_transcript --> {sample_audio_transcript}")
        print(f'created metadata.csv for {dataset}; contains --> {df.shape[0]}, {round(df.duration.sum(),2)} sec')
        del df
        print("\n")

create_metadata_for_indicsuperb_datasets()  
create_metadata_for_openslr_datasets()
create_metadata_for_ulca_datasets()
create_metadata_for_mucs_datasets()

          