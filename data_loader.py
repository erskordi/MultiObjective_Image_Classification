import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Subset
from pathlib import Path

from config import Config


def _shuffle_dataset(dataset, seed=42):
    """Return a shuffled view of a dataset while preserving sample-label pairs."""
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    return Subset(dataset, indices)


def _resolve_split_root(base_root, split_name):
    """Resolve split root, supporting both split/class and split/split/class layouts."""
    split_root = Path(base_root) / split_name
    nested_root = split_root / split_name
    return str(nested_root if nested_root.is_dir() else split_root)


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
    return train_data, test_data, classes

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

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.Grayscale(num_output_channels=1),  # Convert to 1 channel if needed
    ])

    train_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform
    )
    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform
    )
    return train_data, test_data, classes

def gravity_waves_data(return_validation=False):
    """Load gravity waves data from data/1/{train, validation, test}.

    Labels are indexed like FashionMNIST (integer targets with an ordered
    class-name list), and each split is shuffled once after loading.
    """

    # Placeholder for custom dataset loading function
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Grayscale(num_output_channels=1)  # Convert to 1 channel if needed
    ])

    base_root = "data/1"
    train_root = _resolve_split_root(base_root, "train")
    validation_root = _resolve_split_root(base_root, "validation")
    test_root = _resolve_split_root(base_root, "test")

    train_data = datasets.ImageFolder(root=train_root, transform=transform)
    validation_data = datasets.ImageFolder(root=validation_root, transform=transform)
    test_data = datasets.ImageFolder(root=test_root, transform=transform)

    if validation_data.classes != train_data.classes or test_data.classes != train_data.classes:
        raise ValueError(
            "Class folders are inconsistent across train/validation/test splits. "
            "Ensure each split has the same class subfolders."
        )

    # Match FashionMNIST-style labeling: integer targets mapped to ordered class names.
    classes = train_data.classes

    train_data = _shuffle_dataset(train_data, seed=42)
    validation_data = _shuffle_dataset(validation_data, seed=43)
    test_data = _shuffle_dataset(test_data, seed=44)
    
    if return_validation:
        return train_data, validation_data, test_data, classes
    return train_data, test_data, classes

if __name__ == "__main__":
    config = Config()
    data_type = input(f"Select dataset ({', '.join(config.data)}): ").strip()

    if data_type == "FashionMNIST":
        train_data, test_data, classes = load_mnist_data()
    elif data_type == "CIFAR10":
        train_data, test_data, classes = load_cifar10_data()
    elif data_type == "Custom":
        train_data, validation_data, test_data, classes = gravity_waves_data(return_validation=True)
        print(
            f"Loaded custom dataset with {len(train_data)} training samples, "
            f"{len(validation_data)} validation samples, {len(test_data)} test samples, "
            f"and classes: {classes}"
        )
    else:
        raise ValueError(f"Invalid dataset type. Please choose from: {', '.join(config.data)}")