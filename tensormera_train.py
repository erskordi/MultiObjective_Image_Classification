import os
from itertools import product

import torch
from torch import nn
from torch.utils.data import DataLoader
from early_stopping import EarlyStopping

from help_functions import plot_metrics, save_metrics, save_txt, confusion_matrix, auroc, classification_metrics_multiclass
from data_loader import _compute_class_weights
from tensormera import TensorMERA


def train_tensormera(
        model_type, 
        BATCH_SIZE, 
        IMAGE_SIZE, 
        EPOCHS, 
        train_data, 
        test_data, 
        classes, 
        device, 
        monitor, 
        model_dir):
    
    # Grid search over hyperparameters
    # input_dim=32, lambda_reg=0.01 kept at defaults
    grid_search = {
        "num_layers": [2, 4],
        "shrink_factor": [0.7, 0.9],
        "lr": [1e-3, 5e-4],
    }
    
    input_dim = 32  # Default
    lambda_reg = 0.01  # Default

    keys = grid_search.keys()
    values = (grid_search[key] for key in keys)
    combinations = [dict(zip(keys, combination)) for combination in product(*values)]
    
    for combo in combinations:
        print(f"Training {model_type.upper()} with parameters: {combo}")
        num_layers = combo["num_layers"]
        shrink_factor = combo["shrink_factor"]
        lr = combo["lr"]

        tensormera_model = TensorMERA(
            num_classes=len(classes),
            input_channels=train_data[0][0].shape[0],
            input_dim=input_dim,
            num_layers=num_layers,
            shrink_factor=shrink_factor,
            spatial_size=IMAGE_SIZE,
        ).to(device)

        if hasattr(torch, "compile"):
            tensormera_model = torch.compile(tensormera_model)

        # Compute class weights for imbalanced dataset
        class_weights = _compute_class_weights(train_data, len(classes))
        loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = torch.optim.AdamW(tensormera_model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(1, EPOCHS // 3), T_mult=1, eta_min=1e-7
        )

        # Train TensorMERA model
        # Conservative worker count avoids exhausting open file handles during long grid searches.
        num_workers = min(4, os.cpu_count() or 2)
        loader_kwargs = {
            "num_workers": num_workers,
            "pin_memory": (device == "cuda"),
            "persistent_workers": False,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = 2
        
        # Keep imbalance handling in the loss only
        train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, **loader_kwargs)
        test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)
        
        training_loss, accuracy, test_loss, total_energy, mem_utilized, flops = [], [], [], [], [], []
        monitor.begin_window("training")
        early_stopping = EarlyStopping(patience=10)
        mem_before = torch.cuda.memory_allocated() if device == "cuda" else 0
        
        for t in range(EPOCHS + 30):  # Allow extra epochs for potential early stopping   
            print(f"Epoch {t+1}\n-------------------------------")
            total_loss, epoch_energy, epoch_mem = tensormera_model.train_model(
                train_dataloader, loss_fn, optimizer, device, monitor, lambda_reg=lambda_reg
            )
            total_energy.append(epoch_energy)
            mem_utilized.append(epoch_mem)
            acc, avg_loss, avg_flops, _, _, _, _, _ = tensormera_model.test_model(test_dataloader, loss_fn, device)
            early_stopping(avg_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
            scheduler.step()
            training_loss.append(total_loss)
            accuracy.append(acc)
            test_loss.append(avg_loss)
            flops.append(avg_flops)
        
        print("Done!")
        mem_after = torch.cuda.memory_allocated() if device == "cuda" else 0
        measurement = monitor.end_window("training")
        print(f"Entire training: {round(measurement.time,2)} s, {round(measurement.total_energy / 1e6,2)} MJ")
        print(f"Average Loss: {sum(training_loss)/len(training_loss):.4f}, Average Accuracy: {sum(accuracy)/len(accuracy):.4f}, Average Test Loss: {sum(test_loss)/len(test_loss):.4f}")
        print(f"Average Energy: {sum(total_energy)/len(total_energy):.2f} MJ, Average Memory Utilized: {sum(mem_utilized)/len(mem_utilized):.2f} MB")
        print(f"Average FLOPs per sample: {sum(flops)/len(flops):.2f}")
        
        # Save model with parameter configuration in filename
        model_filename = f"tensormera_model_{num_layers}_{shrink_factor}_{lr}.pth"
        tensormera_model.save_model(model_dir / model_filename)

        # Plot training metrics
        plot_metrics(t,
                    training_loss[:t], 
                    accuracy[:t], 
                    total_energy[:t], 
                    model_dir / f"tensormera_metrics_{num_layers}_{shrink_factor}_{lr}.png"
                    )
        
        # Store all metrics in a CSV file for later analysis
        save_metrics(t, 
                    training_loss[:t], 
                    accuracy[:t],
                    test_loss[:t],
                    total_energy[:t],
                    mem_utilized[:t],
                    flops[:t],
                    model_dir / f"tensormera_metrics_{num_layers}_{shrink_factor}_{lr}.csv"
                    )
        save_txt(f"{round((mem_after - mem_before) / 1024**2, 2)}MB\n{sum(flops)/len(flops):.2f}FLOPS", 
                 model_dir / f"tensormera_memory_utilization_{num_layers}_{shrink_factor}.txt")


def evaluate_tensormera_model(model_path, num_layers, shrink_factor, labels, test_data, device, image_size=28):
    """Evaluate the best TensorMERA model and return performance metrics."""
    input_dim = 32  # Default
    tensormera_model = TensorMERA(
        num_classes=len(labels),
        input_channels=test_data[0][0].shape[0],
        input_dim=input_dim,
        num_layers=num_layers,
        shrink_factor=shrink_factor,
        spatial_size=image_size,
    ).to(device)
    tensormera_model.load_state_dict(torch.load(model_path))
    tensormera_model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    test_dataloader = DataLoader(test_data, batch_size=256, shuffle=False)
    _, _, avg_flops, energy, mem_utilization, params, all_preds, all_labels = tensormera_model.test_model(
        test_dataloader, loss_fn, device
    )
    confusion_matrix(all_preds, all_labels, labels, model_path.parent / f"tensormera_confusion_matrix_{model_path.stem}.png")
    auc = auroc(all_preds, all_labels, labels)
    cls_metrics = classification_metrics_multiclass(all_preds.argmax(dim=1), all_labels, labels)
    macro = cls_metrics["macro"]

    return (
        avg_flops,
        energy,
        mem_utilization,
        auc,
        params,
        macro,
    )
