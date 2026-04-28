from pathlib import Path
import numpy as np
import pandas as pd
import torch

from zeus.monitor import ZeusMonitor
from vit import ViTImageClassifier as ViT
from gnn_train import train_gnn
from cnn import CNNImageClassifier as CNN
from cnn_train import train_cnn
from vit_train import train_vit
from data_loader import load_mnist_data, load_cifar10_data, gravity_waves_data
from config import Config
from model_evaluations import evaluations

config = Config()

model_types = config.model_types
data = config.data

def parameter_report(model):
    """Prints the total and trainable parameters of the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

def load_dataset(dataset_type):
    """Loads the specified dataset and returns the training and testing data along with class labels."""
    if dataset_type == data[0]:
        return load_mnist_data()
    elif dataset_type == data[1]:
        return load_cifar10_data()
    elif dataset_type == data[2]:
        return gravity_waves_data()
    else:
        raise ValueError(f"Invalid dataset type. Please choose from: {', '.join(data)}")


if __name__ == "__main__":
    TRAINING = False
    data_type = input(f"Select dataset ({', '.join(data)}): ").strip()
    model_dirs = [Path(f"saved_models/{data_type}/{model_type}") for model_type in model_types]

    if TRAINING:
        print("Training mode enabled. Models will be trained and saved.")
        model_type = input(f"Select model type ({', '.join(model_types)}): ").strip().lower()
        
        # Create directory for saving models and metrics
        for model_dir in model_dirs:
            model_dir.mkdir(exist_ok=True)
        # Reduces file-descriptor pressure from DataLoader shared memory on Linux.
        torch.multiprocessing.set_sharing_strategy("file_system")

    DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {DEVICE} device")

    monitor = ZeusMonitor(gpu_indices=[0])

    train_data, test_data, classes = load_dataset(data_type)

    # Constants
    EPOCHS = 30
    BATCH_SIZE = 256
    IMAGE_SIZE = test_data[0][0].shape[1] 
    

    if TRAINING:
        if model_type == model_types[0]:
            train_vit(ViT, model_type, BATCH_SIZE, IMAGE_SIZE, EPOCHS, train_data, test_data, classes, DEVICE, monitor, model_dirs[0])
        elif model_type == model_types[1]:
            train_cnn(CNN, model_type, BATCH_SIZE, IMAGE_SIZE, EPOCHS, train_data, test_data, classes, DEVICE, monitor, model_dirs[1])
        elif model_type == model_types[2]:
            train_gnn(model_type, BATCH_SIZE, IMAGE_SIZE, EPOCHS, train_data, test_data, classes, DEVICE, monitor, model_dirs[2])
        else:
            print(f"Invalid model type. Please choose from: {', '.join(model_types)}")
    else:
        eval_results = {}
        print("Training skipped. Set training = True to train the model.")
        for model_type in model_types:
            avg_flops, energy, mem_utilization, auc, files = evaluations(model_type, model_dirs, test_data, classes, DEVICE, IMAGE_SIZE)
            eval_results[model_type] = {
                "avg_flops": avg_flops,
                "energy": energy,
                "mem_utilization": mem_utilization,
                "auc": auc,
                "files": files,
            }

        rows = []
        for model_type, metrics in eval_results.items():
            for file_name, flops, en, mem, auc_score in zip(
                metrics["files"],
                metrics["avg_flops"],
                metrics["energy"],
                metrics["mem_utilization"],
                metrics["auc"],
            ):
                rows.append(
                    {
                        "model_type": model_type,
                        "file": file_name,
                        "avg_flops_log": flops,
                        "energy_log": en,
                        "mem_utilization_log": mem,
                        "auc": auc_score,
                    }
                )

        results_df = pd.DataFrame(rows)
        print(results_df)
        results_df.to_csv(f"model_evaluation_results_{data_type}.csv", index=False)