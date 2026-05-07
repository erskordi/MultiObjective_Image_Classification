import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassAUROC

def show_images(batch_tensor, y, classes, n=16, titles=None):
    """
    Display n grayscale images from a batch [B, 1, H, W] in a rectangular grid with optional titles.

    Parameters:
    - batch_tensor: torch.Tensor of shape [B, 1, H, W]
    - y: list or tensor of labels for each image
    - classes: list of class names corresponding to labels
    - n: number of images to show
    - titles: list of titles or labels for each image (length n)
    """
    assert batch_tensor.ndim == 4 and batch_tensor.size(1) == 1, "Expected shape [B, 1, H, W]"
    n = min(n, batch_tensor.size(0))
    if titles is None:
        titles = [f"Image {i}" for i in range(n)]
    else:
        assert len(titles) >= n, "Not enough titles for the number of images."

    # Determine grid size
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    plt.figure(figsize=(cols * 2, rows * 2))
    for i in range(n):
        img = batch_tensor[i, 0].cpu().numpy()
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(classes[y[i]], fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def parameter_report(model):
    print(f"{'Layer':<40} {'# Params':>12} {'Trainable':>10}")
    print("-" * 65)
    total, trainable = 0, 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        is_trainable = param.requires_grad
        print(f"{name:<40} {num_params:>12,} {str(is_trainable):>10}")
        total += num_params
        if is_trainable:
            trainable += num_params
    print("-" * 65)
    print(f"{'Total':<40} {total:>12,} {'':>10}")
    print(f"{'Trainable':<40} {trainable:>12,}")

def plot_metrics(epochs, training_loss, accuracy, total_energy, save_path):
    epochs_axis = list(range(1, epochs + 1))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0, 0].plot(epochs_axis, training_loss, label="Training Loss", color="tab:blue")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs_axis, accuracy, label="Accuracy", color="tab:green")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].set_title("Validation Accuracy")
    axes[0, 1].legend()

    axes[1, 1].plot(epochs_axis, total_energy, label="Energy", color="tab:red")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Energy (J)")
    axes[1, 1].set_title("Energy Per Epoch")
    axes[1, 1].legend()


    axes[1, 0].axis("off")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

def save_metrics(epochs, training_loss, accuracy, test_loss, total_energy, mem_utilized, flops, save_path):
    metrics_df = pd.DataFrame({
                "epoch": list(range(1, epochs+1)),
                "training_loss": training_loss,
                "accuracy": accuracy,
                "test_loss": test_loss,
                "total_energy": total_energy,
                "mem_utilized": mem_utilized,
                "flops": flops
            })
    metrics_df.to_csv(save_path, index=False)

def save_txt(content, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(content)

def confusion_matrix(preds, labels, classes, save_path):
    num_classes = len(classes)
    metric = MulticlassConfusionMatrix(num_classes=num_classes)
    cm_counts = metric(preds.cpu(), labels.cpu()).numpy().astype(np.float64)
    row_sums = cm_counts.sum(axis=1, keepdims=True)
    cm_percent = np.divide(
        cm_counts,
        row_sums,
        out=np.zeros_like(cm_counts, dtype=np.float64),
        where=(row_sums != 0),
    ) * 100.0
    cm_matrix = pd.DataFrame(cm_percent, index=classes, columns=classes)

    fig_width = max(12.0, 0.65 * num_classes)
    fig_height = max(10.0, 0.55 * num_classes)
    tick_fontsize = max(8, min(12, int(220 / max(1, num_classes))))
    annotation_fontsize = max(5, min(10, int(170 / max(1, num_classes))))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(cm_matrix, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=100.0)
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row %")
    tick_marks = range(num_classes)
    ax.set_xticks(list(tick_marks))
    ax.set_yticks(list(tick_marks))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=tick_fontsize)
    ax.set_yticklabels(classes, fontsize=tick_fontsize)

    cm_values = cm_matrix.values
    threshold = cm_values.max() / 2.0 if cm_values.size else 0
    for i in range(cm_values.shape[0]):
        for j in range(cm_values.shape[1]):
            value = cm_values[i, j]
            ax.text(
                j,
                i,
                f"{value:.1f}",
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=annotation_fontsize,
            )

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

def vanilla_cm(preds, labels, classes, save_path):
    cm = torch.zeros((len(classes), len(classes)), dtype=torch.int32)
    for p, t in zip(preds.cpu(), labels.cpu()):
        cm[t, p] += 1
    plt.figure(figsize=(8, 6))
    plt.imshow(cm.numpy(), interpolation='nearest', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def auroc(preds, labels, num_classes):
    auroc_metric = MulticlassAUROC(num_classes=len(num_classes))
    auroc_metric.update(preds.cpu(), labels.cpu())
    return auroc_metric.compute().item()