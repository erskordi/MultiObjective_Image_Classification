
import os
from itertools import product

import torch
from torch import nn
from torch.utils.data import DataLoader
from early_stopping_pytorch import EarlyStopping

from help_functions import plot_metrics, save_metrics, save_txt, confusion_matrix, auroc

def train_vit(
        model, 
        model_type, 
        batch_size, 
        image_size, 
        epochs, 
        train_data, 
        test_data, 
        classes, 
        device, 
        monitor, 
        model_dir):
    # Grid search over hyperparameters
    grid_search = {
            "depth": [6, 12],
            "num_heads": [8, 16],
            "patch_size": [32, 64],
            "lr": [1e-3, 5e-4],
            "patience": [5],
    }

    keys = grid_search.keys()
    values = (grid_search[key] for key in keys)
    combinations = [dict(zip(keys, combination)) for combination in product(*values)]
    for combo in combinations:
        print(f"Training {model_type.upper()} with parameters: {combo}")
        depth = combo["depth"]
        num_heads = combo["num_heads"]
        patch_size = combo["patch_size"]
        lr = combo["lr"]

        vit_model = model(
            img_size=image_size, patch_size=16, in_channels=1, num_classes=len(classes), embed_dim=768,
            depth=depth, num_heads=num_heads, ff_dim=3072, dropout=0.1, use_mdn=False
        ).to(device)

        if hasattr(torch, "compile"):
            vit_model = torch.compile(vit_model)
        #parameter_report(vit_model)
        #summary(vit_model, (1, image_size, image_size))
        

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(vit_model.parameters(), lr=lr, momentum=0.9)

        # Train ViT model
        # Conservative worker count avoids exhausting open file handles during long grid searches.
        num_workers = min(4, os.cpu_count() or 2)
        loader_kwargs = {
            "num_workers": num_workers,
            "pin_memory": (device == "cuda"),
            "persistent_workers": False,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = 2
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **loader_kwargs)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, **loader_kwargs)
        
        training_loss, accuracy, test_loss, total_energy, mem_utilized, flops = [], [], [], [], [], []
        monitor.begin_window("training")
        early_stopping = EarlyStopping(patience=combo["patience"], verbose=True, path=model_dir/f"vit_early_stopped_model_{depth}_{num_heads}_{patch_size}_{lr}.pth")
        mem_before = torch.cuda.memory_allocated() if device == "cuda" else 0
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            total_loss, epoch_energy, epoch_mem = vit_model.train_model(train_dataloader, loss_fn, optimizer, device, monitor)
            total_energy.append(epoch_energy)
            mem_utilized.append(epoch_mem)
            acc, avg_loss, avg_flops, _, _ = vit_model.test_model(test_dataloader, loss_fn, device, monitor)
            early_stopping(avg_loss, vit_model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
            training_loss.append(total_loss)
            accuracy.append(acc)
            test_loss.append(avg_loss)
            flops.append(avg_flops)
        print("Done!")
        mem_after = torch.cuda.memory_allocated() if device == "cuda" else 0
        measurement = monitor.end_window("training")
        print(f"Entire training: {round(measurement.time,2)} s, {round(measurement.total_energy,2)} J")
        print(f"Average Loss: {sum(training_loss)/len(training_loss):.4f}, Average Accuracy: {sum(accuracy)/len(accuracy):.4f}, Average Test Loss: {sum(test_loss)/len(test_loss):.4f}")
        print(f"Average Energy: {sum(total_energy)/len(total_energy):.2f} J, Average Memory Utilized: {sum(mem_utilized)/len(mem_utilized):.2f} MB")
        print(f"Average FLOPs per sample: {sum(flops)/len(flops):.2f}")
        vit_model.save_model(model_dir/f"vit_model_{depth}_{num_heads}_{patch_size}_{lr}.pth")

        # plot training loss, accuracy, energy, memory, and flops
        plot_metrics(t,
                    training_loss[:t], 
                    accuracy[:t], 
                    total_energy[:t], 
                    model_dir/f"vit_metrics_{depth}_{num_heads}_{patch_size}_{lr}.png"
                    )
        
        # store all metrics in a CSV file for later analysis
        save_metrics(t, 
                    training_loss[:t], 
                    accuracy[:t],
                    test_loss[:t],
                    total_energy[:t],
                    mem_utilized[:t],
                    flops[:t],
                    model_dir/f"vit_metrics_{depth}_{num_heads}_{patch_size}_{lr}.csv"
                    )
        save_txt(f"{round((mem_after - mem_before) / 1024**2, 2)}MB\n{sum(flops)/len(flops):.2f}FLOPS", model_dir/f"vit_memory_utilization_{depth}_{num_heads}_{patch_size}_{lr}.txt")

def evaluate_vit_model(
        model_class, 
        model_path, 
        image_size,
        test_data, 
        labels,
        depth, 
        num_heads, 
        patch_size, 
        lr, 
        device):
    vit_model = model_class(
            img_size=image_size, patch_size=16, in_channels=1, num_classes=len(labels), embed_dim=768,
            depth=depth, num_heads=num_heads, ff_dim=3072, dropout=0.1, use_mdn=False
        ).to(device)
    vit_model.load_state_dict(torch.load(model_path))
    vit_model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    test_dataloader = DataLoader(test_data, batch_size=256, shuffle=False)
    _, _, avg_flops, energy, mem_utilization, all_preds, all_labels = vit_model.test_model(test_dataloader, loss_fn, device)
    confusion_matrix(all_preds, all_labels, labels, model_path.parent/f"vit_confusion_matrix_{model_path.stem}.png")
    auc = auroc(all_preds, all_labels, labels)
    
    return avg_flops, energy, mem_utilization, auc