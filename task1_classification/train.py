# task1_classification/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from task1_classification.config import *
from task1_classification.evaluate import validate


def train(model, train_loader, val_loader, device):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):

        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.squeeze().long().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss /= len(train_loader)
        train_acc = correct / total

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(f"Epoch {epoch+1}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            best_val_loss = val_loss

    torch.save(model.state_dict(), FINAL_MODEL_PATH)
