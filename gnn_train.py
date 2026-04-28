import os
from itertools import product
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from early_stopping_pytorch import EarlyStopping
from torch.utils.flop_counter import FlopCounterMode
from torchsummary import summary
from zeus.monitor import ZeusMonitor

from gnn_preprocess import GenericImagePreprocessor, GenericGraphReadyImageDataset
from gnn_graph_build import SparseImageGraphBuilder, SparseGraphReadyDataset, sparse_graph_collate_fn
from help_functions import plot_metrics, save_metrics, save_txt, confusion_matrix, auroc
from pgnn import PGNNConfig, build_pgnn_model


# ============================================================
# Helpers
# ============================================================

def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """
    Moves tensor values in the batch dict to the target device.
    Non-tensor entries are left untouched.
    """
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=(device == "cuda"))
        elif isinstance(v, list):
            out[k] = [item.to(device, non_blocking=(device == "cuda")) if torch.is_tensor(item) else item for item in v]
        else:
            out[k] = v
    return out


@torch.no_grad()
def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def train_model(model, loader, optimizer, criterion, device, monitor=None, scaler=None, use_amp=False):
    """Train the model for one epoch and measure energy and memory usage."""
    model.train()

    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0

    if monitor is not None:
        monitor.begin_window("epoch")
    mem_before = torch.cuda.memory_allocated() if device == "cuda" else 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(
                x1=batch["x1"],
                x2=batch["x2"],
                edge_index=batch["edge_index"],
                batch=batch["batch"],
            )
            loss = criterion(logits, batch["label"])

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        acc = compute_accuracy(logits, batch["label"])

        running_loss += loss.item()
        running_acc += acc
        n_batches += 1

    if monitor is not None:
        measurement = monitor.end_window("epoch")
        epoch_energy = round(measurement.total_energy, 2)
    else:
        epoch_energy = 0.0

    mem_after = torch.cuda.memory_allocated() if device == "cuda" else 0
    mem_utilized = round((mem_after - mem_before) / 1024**2, 2)

    epoch_loss = running_loss / max(1, n_batches)
    epoch_acc = 100.0 * (running_acc / max(1, n_batches))
    return epoch_loss, epoch_acc, epoch_energy, mem_utilized


@torch.no_grad()
def test_model(model, loader, criterion, device, use_amp=False):
    """Evaluate the model on the test dataset and measure energy and memory usage."""
    model.eval()
    n_samples = len(loader.dataset)
    total_flops = 0
    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0

    monitor = ZeusMonitor(gpu_indices=[0])  # Create a monitor instance for testing
    monitor.begin_window("testing")
    mem_before = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            with FlopCounterMode(display=False) as flop_counter:
                logits = model(
                    x1=batch["x1"],
                    x2=batch["x2"],
                    edge_index=batch["edge_index"],
                    batch=batch["batch"],
                )
                all_preds.append(logits.cpu())
                all_labels.append(batch["label"].cpu())
        total_flops += flop_counter.get_total_flops()

        loss = criterion(logits, batch["label"])
        acc = compute_accuracy(logits, batch["label"])

        running_loss += loss.item()
        running_acc += acc
        n_batches += 1

    avg_loss = running_loss / max(1, n_batches)
    avg_acc = 100.0 * (running_acc / max(1, n_batches))
    average_flops = total_flops / max(1, n_samples)
    measurement = monitor.end_window("testing")
    mem_after = torch.cuda.memory_allocated() if device == "cuda" else 0
    mem_utilized = mem_after - mem_before
    print(f"Test Error: \n Accuracy: {avg_acc:>0.1f}%, Avg loss: {avg_loss:>8f} \n")
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    return avg_acc, avg_loss, average_flops, round(measurement.total_energy, 2), round(mem_utilized / 1024**2, 2), all_preds, all_labels


# ============================================================
# Dedicated model training function
# ============================================================

def train_gnn(
    model_type,
    BATCH_SIZE,
    IMAGE_SIZE,
    EPOCHS,
    train_data,
    test_data,
    classes,
    device,
    monitor,
    model_dir,
):
    # Keep IMAGE_SIZE for interface compatibility with other train functions.
    _ = IMAGE_SIZE

    if model_dir is None:
        model_dir = Path("./checkpoints/gnn")
    else:
        model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    grid_search = {
        "x1_hidden_dim": [32, 64],
        "x2_hidden_dim": [32, 64],
        "lr": [1e-3, 5e-4],
    }

    keys = grid_search.keys()
    values = (grid_search[key] for key in keys)
    combinations = [dict(zip(keys, combination)) for combination in product(*values)]

    num_workers = min(4, os.cpu_count() or 2)
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": (device == "cuda"),
        "persistent_workers": False,
        "collate_fn": sparse_graph_collate_fn,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    for combo in combinations:
        print(f"Training {model_type.upper()} with parameters: {combo}")
        x1_hidden_dim = combo["x1_hidden_dim"]
        x2_hidden_dim = combo["x2_hidden_dim"]
        lr = combo["lr"]

        preprocessor = GenericImagePreprocessor(
            normalize=True,
            mean=(0.5,),
            std=(0.5,),
            add_spatial_coords=True,
            include_intensity=True,
            smoothing_kernel_size=3,
        )

        train_pre = GenericGraphReadyImageDataset(
            base_dataset=train_data,
            preprocessor=preprocessor,
            use_patches=False,
        )
        test_pre = GenericGraphReadyImageDataset(
            base_dataset=test_data,
            preprocessor=preprocessor,
            use_patches=False,
        )

        graph_builder = SparseImageGraphBuilder(
            connectivity=4,
            add_self_loops=True,
            undirected=True,
        )

        train_ds = SparseGraphReadyDataset(
            base_dataset=train_pre,
            graph_builder=graph_builder,
            cache_graphs_by_size=True,
        )
        test_ds = SparseGraphReadyDataset(
            base_dataset=test_pre,
            graph_builder=graph_builder,
            cache_graphs_by_size=True,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            **loader_kwargs,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            **loader_kwargs,
        )

        sample_batch = next(iter(train_loader))
        x1_in_dim = sample_batch["x1"].shape[1]
        x2_in_dim = sample_batch["x2"].shape[1]
        num_classes = len(classes)

        print("x1_in_dim:", x1_in_dim)
        print("x2_in_dim:", x2_in_dim)

        cfg = PGNNConfig(
            x1_in_dim=x1_in_dim,
            x2_in_dim=x2_in_dim,
            x1_hidden_dim=x1_hidden_dim,
            x2_hidden_dim=x2_hidden_dim,
            gcn_hidden=64,
            cheb_hidden=64,
            feat_dim=128,
            num_classes=num_classes,
            cheb_k=3,
            dropout=0.1,
            use_layernorm=True,
            fusion="concat",
            cheb_normalization="sym",
        )

        pgnn_model = build_pgnn_model(cfg).to(device)

        optimizer = torch.optim.Adam(
            pgnn_model.parameters(),
            lr=lr,
            weight_decay=1e-4,
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        use_amp = device == "cuda"
        scaler = torch.amp.GradScaler(enabled=use_amp)

        training_loss, accuracy, test_loss = [], [], []
        total_energy, mem_utilized, flops = [], [], []

        if monitor is not None:
            monitor.begin_window("training")

        early_stopping = EarlyStopping(
            patience=7,
            verbose=True,
            path=model_dir / f"gnn_early_stopped_model_{x1_hidden_dim}_{x2_hidden_dim}_{lr}.pth",
        )

        mem_before = torch.cuda.memory_allocated() if device == "cuda" else 0

        for epoch in range(1, EPOCHS + 1):
            tr_loss, tr_acc, epoch_energy, epoch_mem = train_model(
                model=pgnn_model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                monitor=monitor,
                scaler=scaler,
                use_amp=use_amp,
            )

            acc, avg_loss, avg_flops, _, _, _, _ = test_model(
                model=pgnn_model,
                loader=test_loader,
                criterion=criterion,
                device=device,
                use_amp=use_amp,
            )

            training_loss.append(tr_loss)
            accuracy.append(acc)
            test_loss.append(avg_loss)
            total_energy.append(epoch_energy)
            mem_utilized.append(epoch_mem)
            flops.append(avg_flops)

            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.2f}% | "
                f"Test Loss: {avg_loss:.4f}, Test Acc: {acc:.2f}%"
            )

            early_stopping(avg_loss, pgnn_model)
            if early_stopping.early_stop:
                print("Early stopping triggered. Ending training.")
                break

        print("Done!")
        epochs_ran = len(training_loss)
        mem_after = torch.cuda.memory_allocated() if device == "cuda" else 0

        if monitor is not None:
            measurement = monitor.end_window("training")
            print(f"Entire training: {round(measurement.time, 2)} s, {round(measurement.total_energy, 2)} J")

        if epochs_ran > 0:
            print(
                f"Average Loss: {sum(training_loss)/epochs_ran:.4f}, "
                f"Average Accuracy: {sum(accuracy)/epochs_ran:.4f}, "
                f"Average Test Loss: {sum(test_loss)/epochs_ran:.4f}"
            )
            print(
                f"Average Energy: {sum(total_energy)/epochs_ran:.2f} J, "
                f"Average Memory Utilized: {sum(mem_utilized)/epochs_ran:.2f} MB"
            )
            print(f"Average FLOPs per sample: {sum(flops)/epochs_ran:.2f}")

            torch.save(
                pgnn_model.state_dict(),
                model_dir / f"gnn_model_{x1_in_dim}_{x1_hidden_dim}_{x2_in_dim}_{x2_hidden_dim}_{lr}.pth",
            )

            plot_metrics(
                epochs_ran,
                training_loss,
                accuracy,
                total_energy,
                model_dir / f"gnn_metrics_{x1_in_dim}_{x1_hidden_dim}_{x2_in_dim}_{x2_hidden_dim}_{lr}.png",
            )

            save_metrics(
                epochs_ran,
                training_loss,
                accuracy,
                test_loss,
                total_energy,
                mem_utilized,
                flops,
                model_dir / f"gnn_metrics_{x1_in_dim}_{x1_hidden_dim}_{x2_in_dim}_{x2_hidden_dim}_{lr}.csv",
            )

            save_txt(
                f"{round((mem_after - mem_before) / 1024**2, 2)}MB\n{sum(flops)/epochs_ran:.2f}FLOPS",
                model_dir / f"gnn_memory_utilization_{x1_in_dim}_{x1_hidden_dim}_{x2_in_dim}_{x2_hidden_dim}_{lr}.txt",
            )

def evaluate_gnn_model(
        model_class, 
        model_path, 
        test_data, 
        x1_in_param, 
        x1_hidden_param, 
        x2_in_param, 
        x2_hidden_param, 
        lr, 
        labels, 
        device):
    cfg = PGNNConfig(
        x1_in_dim=x1_in_param,
        x2_in_dim=x2_in_param,
        x1_hidden_dim=x1_hidden_param,
        x2_hidden_dim=x2_hidden_param,
        gcn_hidden=64,
        cheb_hidden=64,
        feat_dim=128,
        num_classes=len(labels),
        cheb_k=3,
        dropout=0.1,
        use_layernorm=True,
        fusion="concat",
        cheb_normalization="sym",
    )

    pgnn_model = build_pgnn_model(cfg).to(device)
    pgnn_model.load_state_dict(torch.load(model_path, map_location=device))
    pgnn_model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    preprocessor = GenericImagePreprocessor(
        normalize=True,
        mean=(0.5,),
        std=(0.5,),
        add_spatial_coords=True,
        include_intensity=True,
        smoothing_kernel_size=3,
    )
    test_pre = GenericGraphReadyImageDataset(
        base_dataset=test_data,
        preprocessor=preprocessor,
        use_patches=False,
    )
    graph_builder = SparseImageGraphBuilder(
        connectivity=4,
        add_self_loops=True,
        undirected=True,
    )
    test_ds = SparseGraphReadyDataset(
        base_dataset=test_pre,
        graph_builder=graph_builder,
        cache_graphs_by_size=True,
    )

    num_workers = min(4, os.cpu_count() or 2)
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": (device == "cuda"),
        "persistent_workers": False,
        "collate_fn": sparse_graph_collate_fn,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    test_dataloader = DataLoader(test_ds, batch_size=256, shuffle=False, **loader_kwargs)
    _, _, avg_flops, energy, mem_utilization, all_preds, all_labels = test_model(
        model=pgnn_model,
        loader=test_dataloader,
        criterion=criterion,
        device=device,
        use_amp=(device == "cuda"),
    )
    confusion_matrix(all_preds, all_labels, labels, model_path.parent/f"gnn_confusion_matrix_{model_path.stem}.png")
    auc = auroc(all_preds, all_labels, labels)

    return avg_flops, energy, mem_utilization, auc, x1_in_param, x1_hidden_param, x2_in_param, x2_hidden_param, lr

# ============================================================
# Main
# ============================================================

def main():
    data_root = "./data"
    batch_size = 64
    epochs = 10
    save_dir = Path("./checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    base_transform = transforms.ToTensor()

    full_train_base = datasets.FashionMNIST(
        root=data_root,
        train=True,
        download=True,
        transform=base_transform,
    )

    test_base = datasets.FashionMNIST(
        root=data_root,
        train=False,
        download=True,
        transform=base_transform,
    )

    train_gnn(
        model_type="gnn",
        BATCH_SIZE=batch_size,
        IMAGE_SIZE=28,
        EPOCHS=epochs,
        train_data=full_train_base,
        test_data=test_base,
        classes=[str(i) for i in range(10)],
        device=device,
        monitor=None,
        model_dir=save_dir / "gnn",
    )


if __name__ == "__main__":
    main()
