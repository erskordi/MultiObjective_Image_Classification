class Config:
    """Configuration class to store model types and dataset options."""
    
    def __init__(self):
        from resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
        
        self.model_types = ["vit", "cnn", "gat", "mobilenet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "tensormera"]
        self.data = ["FashionMNIST", "CIFAR10", "GW"]
        self.resnet_map = {
            "resnet18": ResNet18,
            "resnet34": ResNet34,
            "resnet50": ResNet50,
            "resnet101": ResNet101,
            "resnet152": ResNet152,
        }