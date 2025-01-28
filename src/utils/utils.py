import matplotlib.pyplot as plt

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.generation.logits_process import TopKLogitsWarper, LogitsProcessorList

from typing import Dict
import json


def load_model(model_id: str):    
    # processor = AutoProcessor.from_pretrained(model_id)
    # model = AutoModelForImageTextToText.from_pretrained(model_id).to(device)
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, processor, tokenizer

def read_json(file_path: str) -> Dict[str, str]:
    with open(file_path, 'r') as f:
        mapping = json.load(f)
    return mapping
