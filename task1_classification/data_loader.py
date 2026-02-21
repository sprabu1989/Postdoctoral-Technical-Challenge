# task1_classification/data_loader.py

from medmnist import INFO
from torchvision import transforms
from torch.utils.data import DataLoader
import medmnist
from task1_classification.config import BATCH_SIZE


def get_dataloaders():

    data_flag = "pneumoniamnist"
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info["python_class"])

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = DataClass(split="train", transform=train_transform, download=True)
    val_dataset = DataClass(split="val", transform=test_transform, download=True)
    test_dataset = DataClass(split="test", transform=test_transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader
