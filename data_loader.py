from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.transforms import ToTensor

from config import Config


def _resolve_split_root(base_root, split_name):
    """Resolve split root, supporting both split/class and split/split/class layouts."""
    split_root = Path(base_root) / split_name
    nested_root = split_root / split_name
    return str(nested_root if nested_root.is_dir() else split_root)


def _shuffle_dataset(dataset, seed=42):
    """Return a shuffled view of a dataset while preserving sample-label pairs."""
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    return Subset(dataset, indices)




def _compute_class_weights(dataset, num_classes):
    """Compute inverse-frequency class weights for imbalanced dataset.
    
    Weight[c] = total_samples / (num_classes * count[c])
    This upweights minority classes during training.
    
    Args:
        dataset: ImageFolder or Subset with class labels
        num_classes: number of classes
    
    Returns:
        torch.Tensor of shape (num_classes,) with normalized weights
    """
    # Recursively unwrap Subset to find base ImageFolder
    current = dataset
    while isinstance(current, Subset):
        current = current.dataset
    
    base_dataset = current
    
    # Collect all target labels from the original dataset through indexing
    # This works regardless of how many Subset layers wrap it
    targets = []
    def extract_labels(ds, indices=None):
        if isinstance(ds, Subset):
            # Get targets from wrapped dataset and apply subset indices
            sub_targets = extract_labels(ds.dataset, None)
            return [sub_targets[i] for i in ds.indices]
        elif hasattr(ds, 'targets'):
            # ImageFolder has targets attribute
            if indices is not None:
                return [ds.targets[i] for i in indices]
            return ds.targets
        else:
            # Fallback: iterate and extract
            return [y for _, y in ds]
    
    targets = extract_labels(dataset)
    
    # Count samples per class
    class_counts = np.bincount(targets, minlength=num_classes)
    
    # Compute inverse-frequency weights
    total = len(targets)
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        if class_counts[c] > 0:
            weights[c] = total / (num_classes * class_counts[c])
    
    # Normalize to sum to 1 (optional but cleaner)
    weights = weights / weights.sum() * num_classes
    
    return weights


def load_mnist_data():
    """Load FashionMNIST dataset with tensor transform."""
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
        transform=ToTensor(),
    )
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    return train_data, test_data, classes


def load_cifar10_data():
    """Load CIFAR10 dataset with grayscale normalized transform."""
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
        transforms.Grayscale(num_output_channels=1),
    ])

    train_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )
    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )
    return train_data, test_data, classes


def gravity_waves_data(return_validation=False):
    """Load gravity waves data from data/1 using class folders as labels."""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Grayscale(num_output_channels=1),
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
        cli_train_data, cli_test_data, cli_classes = load_mnist_data()
        print(f"Loaded FashionMNIST with {len(cli_train_data)} train / {len(cli_test_data)} test")
    elif data_type == "CIFAR10":
        cli_train_data, cli_test_data, cli_classes = load_cifar10_data()
        print(f"Loaded CIFAR10 with {len(cli_train_data)} train / {len(cli_test_data)} test")
    elif data_type == "Custom":
        cli_train_data, cli_validation_data, cli_test_data, cli_classes = gravity_waves_data(return_validation=True)
        print(
            f"Loaded custom dataset with {len(cli_train_data)} training samples, "
            f"{len(cli_validation_data)} validation samples, {len(cli_test_data)} test samples, "
            f"and classes: {cli_classes}"
        )
    else:
        raise ValueError(f"Invalid dataset type. Please choose from: {', '.join(config.data)}")
