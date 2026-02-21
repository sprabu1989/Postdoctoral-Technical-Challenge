# task1_classification/vit_model.py

import torch
import torch.nn as nn
import timm
from task1_classification.config import MODEL_NAME, NUM_CLASSES


def get_model(device):

    model = timm.create_model(MODEL_NAME, pretrained=True)
    model.head = nn.Linear(model.head.in_features, NUM_CLASSES)
    model = model.to(device)

    return model
