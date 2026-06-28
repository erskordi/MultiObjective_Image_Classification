from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


FAMILY_ORDER = ["cnn", "gat", "vit", "mobilenet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "tensormera"]
FAMILY_DISPLAY = {"cnn": "CNN", "gat": "PGNN", "vit": "ViT", "mobilenet": "MobileNet", "resnet18": "ResNet18", "resnet34": "ResNet34", "resnet50": "ResNet50", "resnet101": "ResNet101", "resnet152": "ResNet152", "tensormera": "TensorMERA"}
PREFIX_BY_FAMILY = {"cnn": "cnn", "gat": "gnn", "vit": "vit", "mobilenet": "mobilenet", "resnet18": "resnet18", "resnet34": "resnet34", "resnet50": "resnet50", "resnet101": "resnet101", "resnet152": "resnet152", "tensormera": "tensormera"}


sns.set_theme(style="darkgrid")


def metric_files(base_dir: Path, dataset: str, family: str) -> list[Path]:
    folder = base_dir / "saved_models" / dataset / family
    if not folder.exists():
        return []
    prefix = PREFIX_BY_FAMILY[family]
    return sorted(folder.glob(f"{prefix}_metrics_*.csv"))


def run_key_from_metric_stem(metric_stem: str, family: str) -> str:
    prefix = PREFIX_BY_FAMILY[family]
    return metric_stem.replace(f"{prefix}_metrics_", f"{prefix}_model_") + ".pth"


def plot_grouped_model_type_metrics(eval_df: pd.DataFrame, out_dir: Path) -> None:
    grouped = eval_df.groupby("model_type")
    for model_type, group in grouped:
        metrics = ["avg_flops", "energy", "mem_utilization", "auc", "params"]
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        axes = axes.flatten()

        for ax, metric in zip(axes, metrics):
            df = group[["file", metric]].sort_values(by=metric, ascending=False)
            sns.barplot(data=df, x="file", y=metric, ax=ax)
            ax.set_title(metric)
            ax.set_xlabel("Model specifics (from filename)")
            ax.set_ylabel(metric)
            ax.tick_params(axis="x", rotation=45)

        for ax in axes[len(metrics):]:
            ax.axis("off")

        fig.suptitle(f"Metrics by Model Type: {model_type}", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(out_dir / f"all_metrics_grouped_by_model_type_{model_type}.png", dpi=180)
        plt.close(fig)


def plot_training_metrics(base_dir: Path, dataset: str, out_dir: Path) -> None:
    fig, axes = plt.subplots(len(FAMILY_ORDER), 2, figsize=(16, 7 * len(FAMILY_ORDER)))
    fig.suptitle(f"Training Metrics - {dataset}", fontsize=22, fontweight="bold", y=0.995)

    for row_idx, family in enumerate(FAMILY_ORDER):
        files = metric_files(base_dir, dataset, family)
        ax_acc = axes[row_idx, 0]
        ax_loss = axes[row_idx, 1]
        title = FAMILY_DISPLAY[family]

        for metrics_file in files:
            mdf = pd.read_csv(metrics_file)
            label = metrics_file.stem
            ax_acc.plot(mdf["epoch"], mdf["accuracy"], linewidth=1.4, alpha=0.9, label=label)
            ax_loss.plot(mdf["epoch"], mdf["training_loss"], linewidth=1.4, alpha=0.9, label=label)

        ax_acc.set_title(f"{title} - Accuracy over Epochs", fontsize=13, fontweight="bold")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy (%)")

        ax_loss.set_title(f"{title} - Training Loss over Epochs", fontsize=13, fontweight="bold")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Training Loss")

        if files:
            ax_acc.legend(fontsize=7, ncol=2, loc="best", frameon=True)
            ax_loss.legend(fontsize=7, ncol=2, loc="best", frameon=True)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / f"{dataset}_training_metrics.png", dpi=180)
    plt.close(fig)


def plot_resource_metrics(base_dir: Path, dataset: str, eval_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(len(FAMILY_ORDER), 3, figsize=(22, 5 * len(FAMILY_ORDER)))
    fig.suptitle(f"Resource Metrics - {dataset}", fontsize=22, fontweight="bold", y=0.995)

    for row_idx, family in enumerate(FAMILY_ORDER):
        files = metric_files(base_dir, dataset, family)
        run_keys = [run_key_from_metric_stem(metrics_file.stem, family) for metrics_file in files]

        family_eval = eval_df[eval_df["model_type"] == family].copy()
        auc_by_file = dict(zip(family_eval["file"].astype(str), family_eval["auc"].astype(float)))
        auc_values = [auc_by_file.get(key, np.nan) for key in run_keys]

        ax_mem = axes[row_idx, 0]
        ax_auc = axes[row_idx, 1]
        ax_flops = axes[row_idx, 2]
        title = FAMILY_DISPLAY[family]

        for metrics_file in files:
            mdf = pd.read_csv(metrics_file)
            label = metrics_file.stem
            ax_mem.plot(mdf["epoch"], mdf["mem_utilized"], linewidth=1.3, alpha=0.9, label=label)
            ax_flops.plot(mdf["epoch"], mdf["flops"], linewidth=1.3, alpha=0.9, label=label)

        ax_auc.plot(range(1, len(auc_values) + 1), auc_values, marker="o", linewidth=1.5, markersize=4)

        ax_mem.set_title(f"{title} - Memory Utilization over Epochs", fontsize=12, fontweight="bold")
        ax_mem.set_xlabel("Epoch")
        ax_mem.set_ylabel("Memory Utilization (MB)")

        ax_auc.set_title(f"{title} - AUC per Run", fontsize=12, fontweight="bold")
        ax_auc.set_xlabel("Model Run Index")
        ax_auc.set_ylabel("AUC")

        ax_flops.set_title(f"{title} - FLOPs over Epochs", fontsize=12, fontweight="bold")
        ax_flops.set_xlabel("Epoch")
        ax_flops.set_ylabel("FLOPs")

        if files:
            ax_mem.legend(fontsize=6.5, ncol=2, loc="best", frameon=True)
            ax_flops.legend(fontsize=6.5, ncol=2, loc="best", frameon=True)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / f"{dataset}_memory_auc_flops.png", dpi=180)
    plt.close(fig)


def regenerate_dataset_figures(base_dir: Path, dataset: str) -> None:
    eval_csv = base_dir / f"model_evaluation_results_{dataset}.csv"
    if not eval_csv.exists():
        print(f"[skip] Missing evaluation CSV: {eval_csv}")
        return

    eval_df = pd.read_csv(eval_csv)
    out_dir = base_dir / "evaluation_results" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_grouped_model_type_metrics(eval_df, out_dir)
    plot_training_metrics(base_dir, dataset, out_dir)
    plot_resource_metrics(base_dir, dataset, eval_df, out_dir)

    print(f"[ok] Updated figures for {dataset} in {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate evaluation figures from saved model metrics and evaluation CSV files."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["FashionMNIST", "CIFAR10", "GW"],
        help="Datasets to regenerate (default: FashionMNIST CIFAR10 GW)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project directory containing saved_models/ and evaluation_results/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()

    for dataset in args.datasets:
        regenerate_dataset_figures(base_dir, dataset)

    print("[done] Figure regeneration complete.")


if __name__ == "__main__":
    main()
