# task3_biomedclip_retrieval/data_loader.py

from medmnist import PneumoniaMNIST
from torchvision import transforms


def get_dataset():

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
    ])

    dataset = PneumoniaMNIST(
        split="test",
        download=True,
        transform=transform
    )

    return dataset
