import numpy as np
from config import Config
from gnn_train import evaluate_gnn_model
from cnn_train import evaluate_cnn_model
from vit_train import evaluate_vit_model

from vit import ViTImageClassifier as ViT
from cnn import CNNImageClassifier as CNN

def evaluations(model_type, model_dirs, test_data, classes, device, image_size):
    """Evaluates all saved models of the specified type and returns their performance metrics."""
    config = Config()
    if model_type == config.model_types[0]:  # ViT
        flops, en, mem, area, files = [], [], [], [], []
        for file in model_dirs[0].glob("*.pth"):
            if file.stem.split("_")[1] != "early":
                print(f"Testing ViT model: {file.name}")
                depth, num_heads, patch_size = map(int, file.stem.split("_")[2:5])
                lr = float(file.stem.split("_")[5])
                print(f"Model parameters - Depth: {depth}, Num Heads: {num_heads}, Patch Size: {patch_size}, Learning Rate: {lr}")
                avg_flops, energy, mem_utilization, auc, depth, num_heads, patch_size, lr = evaluate_vit_model(ViT, file, image_size, test_data, classes, depth, num_heads, patch_size, lr, device)
                flops.append(avg_flops)
                en.append(energy)
                mem.append(mem_utilization)
                area.append(auc)
                files.append(file.name)
        return np.log(flops), np.log(en), np.log(mem), area, files
    elif model_type == config.model_types[1]:  # CNN
        flops, en, mem, area, files = [], [], [], [], []
        for file in model_dirs[1].glob("*.pth"):
            if file.stem.split("_")[1] != "early":
                print(f"Testing CNN model: {file.name}")
                output_channels, num_conv_layers, kernel_size = map(int, file.stem.split("_")[2:5])
                lr = float(file.stem.split("_")[5])
                print(f"Model parameters - Output Channels: {output_channels}, Num Conv Layers: {num_conv_layers}, Kernel Size: {kernel_size}, Learning Rate: {lr}")
                avg_flops, energy, mem_utilization, auc, output_channels, num_conv_layers, kernel_size, lr = evaluate_cnn_model(CNN, file, output_channels, num_conv_layers, kernel_size, lr, classes, test_data, device)
                flops.append(avg_flops)
                en.append(energy)
                mem.append(mem_utilization)
                area.append(auc)
                files.append(file.name)
        return np.log(flops), np.log(en), np.log(mem), area, files
    elif model_type == config.model_types[2]:  # GNN
        flops, en, mem, area, files = [], [], [], [], []
        for file in model_dirs[2].glob("*.pth"):
            if file.stem.split("_")[1] != "early":
                print(f"Testing GNN model: {file.name}")
                x1_in_param, x1_hidden_param, x2_in_param, x2_hidden_param = map(int, file.stem.split("_")[2:6])
                lr = float(file.stem.split("_")[6])
                print(f"Model parameters - X1 In: {x1_in_param}, X1 Hidden: {x1_hidden_param}, X2 In: {x2_in_param}, X2 Hidden: {x2_hidden_param}, Learning Rate: {lr}")
                avg_flops, energy, mem_utilization, auc, x1_in_param, x1_hidden_param, x2_in_param, x2_hidden_param, lr = evaluate_gnn_model(model_type, file, test_data, x1_in_param, x1_hidden_param, x2_in_param, x2_hidden_param, lr, classes, device)
                flops.append(avg_flops)
                en.append(energy)
                mem.append(mem_utilization)
                area.append(auc)
                files.append(file.name)
        return np.log(flops), np.log(en), np.log(mem), area, files