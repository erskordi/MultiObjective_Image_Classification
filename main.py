from pathlib import Path
import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
import seaborn as sns

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

def evalResults(model_types, model_dirs, data_type, test_data, classes, device, image_size):
    eval_results = {}
    print("Training skipped. Set training = True to train the model.")
    for model_type in model_types:
        avg_flops, energy, mem_utilization, auc,  files, params = evaluations(model_type, model_dirs, test_data, classes, device, image_size)
        eval_results[model_type] = {
            "avg_flops": avg_flops,
            "energy": energy,
            "mem_utilization": mem_utilization,
            "params": params,
            "auc": auc,
            "files": files,
        }

    rows = []
    for model_type, metrics in eval_results.items():
        for file_name, flops, en, mem, params, auc_score in zip(
            metrics["files"],
            metrics["avg_flops"],
            metrics["energy"],
            metrics["mem_utilization"],
            metrics["params"],
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
                    "params": params,
                }
            )

    results_df = pd.DataFrame(rows)
    print(results_df)
    results_df.to_csv(f"model_evaluation_results_{data_type}.csv", index=False)

    return results_df

if __name__ == "__main__":
    TRAINING = False
    data_type = input(f"Select dataset ({', '.join(data)}): ").strip()
    model_dirs = [Path(f"saved_models/{data_type}/{model_type}") for model_type in model_types]

    # Create directories for saving models and metrics
    for model_dir in model_dirs:
        model_dir.mkdir(parents=True, exist_ok=True)

    if TRAINING:
        print("Training mode enabled. Models will be trained and saved.")
        model_type = input(f"Select model type ({', '.join(model_types)}): ").strip().lower()
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
        results_df = evalResults(model_types, model_dirs, data_type, test_data, classes, DEVICE, IMAGE_SIZE)

        # group by model type, sort by each metric, and plot bar charts for FLOPs, energy, memory, and AUC
        grouped_results = results_df.groupby("model_type")

        results_dir = Path(f"evaluation_results/{data_type}")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        for model_type, group in grouped_results:
            print(f"\nModel Type: {model_type}")
            metrics = ["avg_flops_log", "energy_log", "mem_utilization_log", "auc", "params"]
            fig, axes = plt.subplots(3, 2, figsize=(16, 14))
            axes = axes.flatten()

            for ax, metric in zip(axes, metrics):
                df = group[["file", metric]].sort_values(by=metric, ascending=False)
                print(df)
                sns.barplot(data=df, x="file", y=metric, ax=ax)
                ax.set_title(metric)
                ax.set_xlabel("Model specifics (from filename)")
                ax.set_ylabel(metric)
                ax.tick_params(axis="x", rotation=45)

            for ax in axes[len(metrics):]:
                ax.axis("off")

            fig.suptitle(f"Metrics by Model Type: {model_type}", fontsize=16)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            plt.savefig(results_dir / f"all_metrics_grouped_by_model_type_{model_type}.png")
            plt.close()