# task2_report_generation/medgemma_model.py

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from task2_report_generation.config import MODEL_ID


def load_model(device):

    processor = AutoProcessor.from_pretrained(MODEL_ID, token=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        device_map="auto",
        token=True
    )

    model.eval()
    return processor, model
