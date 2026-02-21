# task3_biomedclip_retrieval/model_loader.py

import torch
from open_clip import create_model_from_pretrained
from task3_biomedclip_retrieval.config import MODEL_NAME


def load_model(device):

    model, preprocess = create_model_from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()

    return model, preprocess
