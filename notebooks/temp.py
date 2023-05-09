import requests
import json
import base64
import ast

def wav_to_base64string(file_path):
    return base64.b64encode(open(file_path, "rb").read()).decode('utf-8') url = "https://ai4b-dev-asr.ulcacontrib.org/asr/v1/recognize/te" 

def get_payload(base64_audio_content):
    payload = json.dumps({
        "config": {
            "language": {
                "sourceLanguage": "te"},
                "transcriptionFormat": {
                    "value": "transcript"},
                    "audioFormat": "wav"
                    },
                    "audio": [{"audioContent": f"{base64_audio_content}"}]})
    return payload

headers = {'Content-Type': 'application/json'}