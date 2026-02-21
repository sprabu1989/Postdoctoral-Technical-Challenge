# task3_biomedclip_retrieval/embedder.py

import torch
import numpy as np
from tqdm import tqdm
from task3_biomedclip_retrieval.config import BATCH_SIZE


def extract_embeddings(model, preprocess, dataset, device):

    embeddings = []
    labels = []
    image_store = []

    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), BATCH_SIZE)):

            batch_imgs = []
            batch_labels = []

            for j in range(i, min(i + BATCH_SIZE, len(dataset))):
                img, lbl = dataset[j]
                batch_imgs.append(preprocess(img))
                batch_labels.append(lbl.item())

            imgs_tensor = torch.stack(batch_imgs).to(device)

            image_features = model.encode_image(imgs_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            embeddings.append(image_features.cpu().numpy())
            labels.extend(batch_labels)
            image_store.extend(imgs_tensor.cpu())

    embeddings = np.vstack(embeddings).astype("float32")
    labels = np.array(labels).astype(int)

    return embeddings, labels, image_store
