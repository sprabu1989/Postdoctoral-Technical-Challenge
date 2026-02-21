# task3_clip_finetuned_retrieval/model_loader.py

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from task3_clip_finetuned_retrieval.config import MODEL_NAME


def load_model(device, num_classes):

    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    classifier = nn.Linear(
        model.config.projection_dim,
        num_classes
    ).to(device)

    return model, processor, classifier
