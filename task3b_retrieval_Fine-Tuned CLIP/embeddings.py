# task3_clip_finetuned_retrieval/embeddings.py

import torch
import numpy as np
from tqdm import tqdm
from task3_clip_finetuned_retrieval.finetune import get_image_embeddings


def extract_test_embeddings(model, processor, test_dataset, device):

    model.eval()

    embeddings = []
    labels = []

    for img, label in tqdm(test_dataset):

        img = img.unsqueeze(0).to(device)

        inputs = processor(
            images=img,
            return_tensors="pt",
            do_rescale=False
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            features = get_image_embeddings(
                model,
                inputs["pixel_values"]
            )

        features = features / features.norm(
            p=2,
            dim=-1,
            keepdim=True
        )

        embeddings.append(features.cpu().numpy())
        labels.append(label.item())

    embeddings = np.vstack(embeddings).astype("float32")
    labels = np.array(labels)

    return embeddings, labels
