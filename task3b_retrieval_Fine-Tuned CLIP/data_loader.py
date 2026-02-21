# task3_clip_finetuned_retrieval/data_loader.py

from medmnist import INFO, PneumoniaMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from task3_clip_finetuned_retrieval.config import BATCH_SIZE, DATA_FLAG


def get_dataloaders():

    info = INFO[DATA_FLAG]

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])

    train_dataset = PneumoniaMNIST(
        split="train",
        transform=transform,
        download=True
    )

    test_dataset = PneumoniaMNIST(
        split="test",
        transform=transform,
        download=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    return train_loader, test_dataset, info
