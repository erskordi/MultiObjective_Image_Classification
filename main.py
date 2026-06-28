from pathlib import Path

import pandas as pd
import torch

import matplotlib.pyplot as plt
import seaborn as sns

from zeus.monitor import ZeusMonitor
from config import Config
from data_loader import load_mnist_data, load_cifar10_data, gravity_waves_data
from model_evaluations import evaluations

from vit import ViTImageClassifier as ViT
from cnn import CNNImageClassifier as CNN
from mobilenet import MobileNetV3Large as MobileNet

from vit_train import train_vit
from cnn_train import train_cnn
from gnn_train import train_gnn
from mobilenet_train import train_mobilenet
from resnet_train import train_resnet
from tensormera_train import train_tensormera


config = Config()
model_types = config.model_types
data = config.data

# Central training/dataset knobs.
EPOCHS = 200
BATCH_SIZE = 256

_RESNET_MODEL_MAP = config.resnet_map

def load_dataset(dataset_type: str):
    """Load the selected dataset and return train/test sets and class labels."""
    if dataset_type == "FashionMNIST":
        return load_mnist_data()
    if dataset_type == "CIFAR10":
        return load_cifar10_data()
    if dataset_type == "GW":
        return gravity_waves_data()
    raise ValueError(f"Invalid dataset type. Please choose from: {', '.join(data)}")


def eval_results(
    configured_model_types,
    configured_model_dirs,
    selected_data_type,
    eval_test_data,
    eval_classes,
    eval_device,
    eval_image_size,
):
    """Run evaluation for all configured model types and write the dataset CSV."""
    eval_out = {}
    print("Training skipped. Set training = True to train the model.")

    for model_name in configured_model_types:
        avg_flops, energy, mem_utilization, auc, files, params = evaluations(
            model_name,
            configured_model_dirs,
            eval_test_data,
            eval_classes,
            eval_device,
            eval_image_size,
        )
        eval_out[model_name] = {
            "avg_flops": avg_flops,
            "energy": energy,
            "mem_utilization": mem_utilization,
            "params": params,
            "auc": auc,
            "files": files,
        }

    rows = []
    for model_name, metric_bundle in eval_out.items():
        for file_name, flops, en, mem, params, auc_score in zip(
            metric_bundle["files"],
            metric_bundle["avg_flops"],
            metric_bundle["energy"],
            metric_bundle["mem_utilization"],
            metric_bundle["params"],
            metric_bundle["auc"],
        ):
            rows.append(
                {
                    "model_type": model_name,
                    "file": file_name,
                    "avg_flops": flops,
                    "energy": en,
                    "mem_utilization": mem,
                    "auc": auc_score,
                    "params": params,
                }
            )

    eval_results_df = pd.DataFrame(rows)
    print(eval_results_df)
    eval_results_df.to_csv(f"model_evaluation_results_{selected_data_type}.csv", index=False)
    return eval_results_df


if __name__ == "__main__":
    training = bool(
        int(
            input(
                "Enter 1 to enable training mode (train and save models), "
                "or 0 to skip training and only evaluate existing models: "
            ).strip()
        )
    )

    data_type = input(f"Select dataset ({', '.join(data)}): ").strip()
    model_dirs = [Path(f"saved_models/{data_type}/{model_type}") for model_type in model_types]

    for model_dir in model_dirs:
        model_dir.mkdir(parents=True, exist_ok=True)

    if training:
        print("Training mode enabled. Models will be trained and saved.")
        model_type = input(f"Select model type ({', '.join(model_types)}): ").strip().lower()
        torch.multiprocessing.set_sharing_strategy("file_system")

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    monitor = ZeusMonitor(gpu_indices=[0])
    train_data, test_data, classes = load_dataset(data_type)
    image_size = test_data[0][0].shape[1]

    if training:
        if model_type == "vit":
            train_vit(
                ViT,
                model_type,
                BATCH_SIZE,
                image_size,
                EPOCHS,
                train_data,
                test_data,
                classes,
                device,
                monitor,
                model_dirs[model_types.index("vit")],
            )
        elif model_type == "cnn":
            train_cnn(
                CNN,
                model_type,
                BATCH_SIZE,
                image_size,
                EPOCHS,
                train_data,
                test_data,
                classes,
                device,
                monitor,
                model_dirs[model_types.index("cnn")],
            )
        elif model_type == "gat":
            train_gnn(
                model_type,
                BATCH_SIZE,
                image_size,
                EPOCHS,
                train_data,
                test_data,
                classes,
                device,
                monitor,
                model_dirs[model_types.index("gat")],
            )
        elif model_type == "mobilenet":
            train_mobilenet(
                MobileNet,
                model_type,
                BATCH_SIZE,
                image_size,
                EPOCHS,
                train_data,
                test_data,
                classes,
                device,
                monitor,
                model_dirs[model_types.index("mobilenet")],
            )
        elif model_type in _RESNET_MODEL_MAP:
            train_resnet(
                _RESNET_MODEL_MAP[model_type],
                model_type,
                BATCH_SIZE,
                image_size,
                EPOCHS,
                train_data,
                test_data,
                classes,
                device,
                monitor,
                model_dirs[model_types.index(model_type)],
            )
        elif model_type == "tensormera":
            train_tensormera(
                model_type,
                BATCH_SIZE,
                image_size,
                EPOCHS,
                train_data,
                test_data,
                classes,
                device,
                monitor,
                model_dirs[model_types.index("tensormera")],
            )
        else:
            print(f"Invalid model type. Please choose from: {', '.join(model_types)}")
    else:
        eval_df = eval_results(
            model_types,
            model_dirs,
            data_type,
            test_data,
            classes,
            device,
            image_size,
        )

        try:
            grouped_results = eval_df.groupby("model_type")
            results_dir = Path(f"evaluation_results/{data_type}")
            results_dir.mkdir(parents=True, exist_ok=True)

            for model_type, group in grouped_results:
                print(f"\nModel Type: {model_type}")
                metric_names = ["avg_flops", "energy", "mem_utilization", "auc", "params"]
                fig, axes = plt.subplots(3, 2, figsize=(16, 14))
                axes = axes.flatten()

                for ax, metric in zip(axes, metric_names):
                    df = group[["file", metric]].sort_values(by=metric, ascending=False)
                    print(df)
                    sns.barplot(data=df, x="file", y=metric, ax=ax)
                    ax.set_title(metric)
                    ax.set_xlabel("Model specifics (from filename)")
                    ax.set_ylabel(metric)
                    ax.tick_params(axis="x", rotation=45)

                for ax in axes[len(metric_names):]:
                    ax.axis("off")

                fig.suptitle(f"Metrics by Model Type: {model_type}", fontsize=16)
                fig.tight_layout(rect=[0, 0, 1, 0.97])
                plt.savefig(results_dir / f"all_metrics_grouped_by_model_type_{model_type}.png")
                plt.close()
        except (ValueError, KeyError) as exc:
            print(
                "Error:",
                exc,
                "\tProbably no evaluation results found. "
                "Please run with training mode enabled to train models and generate evaluation results.",
            )
