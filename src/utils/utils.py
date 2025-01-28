import matplotlib.pyplot as plt

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import TopKLogitsWarper, LogitsProcessorList

from typing import Dict
import json


def load_model(model_id: str):   
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, processor, tokenizer

def read_json(file_path: str) -> Dict[str, str]:
    with open(file_path, 'r') as f:
        mapping = json.load(f)
    return mapping
