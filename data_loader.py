import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import ToTensor

import pandas as pd


def load_mnist_data():
    """Loading FashionMNIST dataset using torchvision.datasets and applying necessary transformations."""
    
    # Reduces file-descriptor pressure from DataLoader shared memory on Linux.
    torch.multiprocessing.set_sharing_strategy("file_system")

    
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    return train_data, test_data, classes\

def load_cifar10_data():
    """Loading CIFAR10 dataset using torchvision.datasets and applying necessary transformations."""

    # Reduces file-descriptor pressure from DataLoader shared memory on Linux.
    torch.multiprocessing.set_sharing_strategy("file_system")

    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    train_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    return train_data, test_data, classes

def gravity_waves_data():
    """Load gravity waves dataset from local directory. Assumes a folder structure of data/1/{train, validation, test} with images and a metadata.csv file containing labels."""

    # Placeholder for custom dataset loading function
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_data = datasets.ImageFolder(root="data/1/train", transform=transform)
    validation_data = datasets.ImageFolder(root="data/1/validation", transform=transform)
    test_data = datasets.ImageFolder(root="data/1/test", transform=transform)

    metadata = pd.read_csv("data/1/metadata.csv")
    classes = metadata["label"].unique().tolist()
    
    return train_data, validation_data, test_data, classes