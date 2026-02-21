# task2_report_generation/data_loader.py

from medmnist import INFO, PneumoniaMNIST
from torchvision import transforms


def get_datasets():

    data_flag = "pneumoniamnist"
    info = INFO[data_flag]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = PneumoniaMNIST(split="train", transform=transform, download=True)
    val_dataset = PneumoniaMNIST(split="val", transform=transform, download=True)
    test_dataset = PneumoniaMNIST(split="test", transform=transform, download=True)

    return train_dataset, val_dataset, test_dataset
