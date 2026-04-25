
import os
from itertools import product

import torch
from torch import nn
from torch.utils.data import DataLoader
from early_stopping_pytorch import EarlyStopping

from help_functions import plot_metrics, save_metrics, save_txt, confusion_matrix, auroc
from pathlib import Path

def train_cnn(
        CNN, 
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
    
    grid_search = {
        "output_channels": [32, 64],
        "num_conv_layers": [2, 3, 4],
        "kernel_size": [3],
        "lr": [1e-3, 5e-4],
    }

    keys = grid_search.keys()
    values = (grid_search[key] for key in keys)
    combinations = [dict(zip(keys, combination)) for combination in product(*values)]
    for combo in combinations:
        print(f"Training {model_type.upper()} with parameters: {combo}")
        output_channels = combo["output_channels"]
        num_conv_layers = combo["num_conv_layers"]
        kernel_size = combo["kernel_size"]
        lr = combo["lr"]

        cnn_model = CNN(
            num_classes=len(classes),
            input_channels=test_data[0][0].shape[0],
            output_channels=output_channels,
            kernel_size=kernel_size,
            num_conv_layers=num_conv_layers
        ).to(device)


        if hasattr(torch, "compile"):
            cnn_model = torch.compile(cnn_model)
        #parameter_report(cnn_model)
        #summary(cnn_model, (1, 28, 28))

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(cnn_model.parameters(), lr=lr, momentum=0.9)

        # Train CNN model
        # Conservative worker count avoids exhausting open file handles during long grid searches.
        num_workers = min(4, os.cpu_count() or 2)
        loader_kwargs = {
            "num_workers": num_workers,
            "pin_memory": (device == "cuda"),
            "persistent_workers": False,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = 2
        train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, **loader_kwargs)
        test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)
        
        training_loss, accuracy, test_loss, total_energy, mem_utilized, flops = [], [], [], [], [], []
        monitor.begin_window("training")
        early_stopping = EarlyStopping(patience=7, verbose=True, path=model_dir/f"cnn_early_stopped_model_{output_channels}_{num_conv_layers}_{kernel_size}.pth")
        mem_before = torch.cuda.memory_allocated() if device == "cuda" else 0
        for t in range(EPOCHS+30):  # Allow extra epochs for potential early stopping   
            print(f"Epoch {t+1}\n-------------------------------")
            total_loss, epoch_energy, epoch_mem = cnn_model.train_model(train_dataloader, loss_fn, optimizer, device, monitor)
            total_energy.append(epoch_energy)
            mem_utilized.append(epoch_mem)
            acc, avg_loss, avg_flops, _, _ = cnn_model.test_model(test_dataloader, loss_fn, device)
            early_stopping(avg_loss, cnn_model)
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
        cnn_model.save_model(model_dir/f"cnn_model_{output_channels}_{num_conv_layers}_{kernel_size}_{lr}.pth")

        # plot training loss, accuracy, energy, memory, and flops
        plot_metrics(t,
                    training_loss[:t], 
                    accuracy[:t], 
                    total_energy[:t], 
                    model_dir/f"cnn_metrics_{output_channels}_{num_conv_layers}_{kernel_size}_{lr}.png"
                    )
        
        # store all metrics in a CSV file for later analysis
        save_metrics(t, 
                    training_loss[:t], 
                    accuracy[:t],
                    test_loss[:t],
                    total_energy[:t],
                    mem_utilized[:t],
                    flops[:t],
                    model_dir/f"cnn_metrics_{output_channels}_{num_conv_layers}_{kernel_size}_{lr}.csv"
                    )
        save_txt(f"{round((mem_after - mem_before) / 1024**2, 2)}MB\n{sum(flops)/len(flops):.2f}FLOPS", model_dir/f"cnn_memory_utilization_{output_channels}_{num_conv_layers}_{kernel_size}.txt")

def evaluate_cnn_model(model_class, model_dir, output_channels, num_conv_layers, kernel_size, lr, labels, test_data, device):
    """Evaluate the best CNN model based on early stopping and return performance metrics."""
    
    # Load the best model based on early stopping
    best_model_path = model_dir.glob("cnn_early_stopped_model_*.pth")
    best_model_path = max(best_model_path, key=os.path.getctime)  # Get the most recently saved model
    print(f"Evaluating model: {best_model_path.name}")
    cnn_model = model_class(
        num_classes=len(labels),
        input_channels=test_data[0][0].shape[0],
        output_channels=output_channels,
        kernel_size=kernel_size,
        num_conv_layers=num_conv_layers
    ).to(device)
    cnn_model.load_state_dict(torch.load(best_model_path))
    loss_fn = nn.CrossEntropyLoss()
    test_dataloader = DataLoader(test_data, batch_size=256, shuffle=False)
    _, _, avg_flops, energy, mem_utilization, all_preds, all_labels = cnn_model.test_model(test_dataloader, loss_fn, device)
    confusion_matrix(all_preds, all_labels, labels, best_model_path.parent/f"cnn_confusion_matrix_{best_model_path.stem}.png")
    auc = auroc(all_preds, all_labels, labels)
    
    return avg_flops, energy, mem_utilization, auc