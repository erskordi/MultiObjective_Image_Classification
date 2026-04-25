class Config:
    """Configuration class to store model types and dataset options."""
    def __init__(self):
        self.model_types = ["vit", "cnn", "gat"]
        self.data = ["FashionMNIST", "CIFAR10", "Custom"]