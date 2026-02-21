# task3_clip_finetuned_retrieval/finetune.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from task3_clip_finetuned_retrieval.config import EPOCHS, LR


def get_image_embeddings(model, images):
    vision_outputs = model.vision_model(pixel_values=images)
    pooled_output = vision_outputs.pooler_output
    projected = model.visual_projection(pooled_output)
    return projected


def fine_tune(model, processor, classifier, train_loader, device):

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        list(model.parameters()) + list(classifier.parameters()),
        lr=LR
    )

    model.train()
    classifier.train()

    for epoch in range(EPOCHS):
        total_loss = 0

        for imgs, labels in tqdm(train_loader):

            imgs = imgs.to(device)
            labels = labels.squeeze().long().to(device)

            inputs = processor(
                images=imgs,
                return_tensors="pt",
                do_rescale=False
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            image_features = get_image_embeddings(
                model,
                inputs["pixel_values"]
            )

            image_features = image_features / image_features.norm(
                p=2,
                dim=-1,
                keepdim=True
            )

            logits = classifier(image_features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

    print("Fine-tuning complete.")
