from config import Config
from gnn_train import evaluate_gnn_model
from cnn_train import evaluate_cnn_model
from vit_train import evaluate_vit_model
from mobilenet_train import evaluate_mobilenet_model
from resnet_train import evaluate_resnet_model
from tensormera_train import evaluate_tensormera_model

from vit import ViTImageClassifier as ViT
from cnn import CNNImageClassifier as CNN
from mobilenet import MobileNetV3Large as MobileNet

config = Config()
_RESNET_MODEL_MAP = config.resnet_map

def evaluations(model_type, model_dirs, test_data, classes, device, image_size):
    """Evaluates all saved models of the specified type and returns their performance metrics."""
    cfg = Config()
    if model_type == cfg.model_types[0]:  # ViT
        flops, en, mem, area, params, files = [], [], [], [], [], []

        for file in model_dirs[0].glob("*.pth"):
            if file.stem.split("_")[1] != "early":
                print(f"Testing ViT model: {file.name}")
                depth, num_heads, patch_size = map(int, file.stem.split("_")[2:5])
                lr = float(file.stem.split("_")[5])
                print(f"Model parameters - Depth: {depth}, Num Heads: {num_heads}, Patch Size: {patch_size}, Learning Rate: {lr}")
                avg_flops, energy, mem_utilization, auc, depth, num_heads, patch_size, lr, num_parameters = evaluate_vit_model(ViT, file, image_size, test_data, classes, depth, num_heads, patch_size, lr, device)
                flops.append(avg_flops)
                en.append(energy)
                mem.append(mem_utilization)
                area.append(auc)
                params.append(num_parameters)
                files.append(file.name)
        return flops, en, mem, area, files, params
    elif model_type == cfg.model_types[1]:  # CNN
        flops, en, mem, area, params, files = [], [], [], [], [], []

        for file in model_dirs[1].glob("*.pth"):
            if file.stem.split("_")[1] != "early":
                print(f"Testing CNN model: {file.name}")
                output_channels, num_conv_layers, kernel_size = map(int, file.stem.split("_")[2:5])
                lr = float(file.stem.split("_")[5])
                print(f"Model parameters - Output Channels: {output_channels}, Num Conv Layers: {num_conv_layers}, Kernel Size: {kernel_size}, Learning Rate: {lr}")
                avg_flops, energy, mem_utilization, auc, output_channels, num_conv_layers, kernel_size, lr, num_parameters = evaluate_cnn_model(CNN, file, output_channels, num_conv_layers, kernel_size, lr, classes, test_data, device)
                flops.append(avg_flops)
                en.append(energy)
                mem.append(mem_utilization)
                area.append(auc)
                params.append(num_parameters)
                files.append(file.name)
        return flops, en, mem, area, files, params
    elif model_type == cfg.model_types[2]:  # GNN
        flops, en, mem, area, params, files = [], [], [], [], [], []

        for file in model_dirs[2].glob("*.pth"):
            if file.stem.split("_")[1] != "early":
                print(f"Testing GNN model: {file.name}")
                x1_in_param, x1_hidden_param, x2_in_param, x2_hidden_param = map(int, file.stem.split("_")[2:6])
                lr = float(file.stem.split("_")[6])
                print(f"Model parameters - X1 In: {x1_in_param}, X1 Hidden: {x1_hidden_param}, X2 In: {x2_in_param}, X2 Hidden: {x2_hidden_param}, Learning Rate: {lr}")
                avg_flops, energy, mem_utilization, auc, x1_in_param, x1_hidden_param, x2_in_param, x2_hidden_param, lr, num_parameters = evaluate_gnn_model(model_type, file, test_data, x1_in_param, x1_hidden_param, x2_in_param, x2_hidden_param, lr, classes, device)
                flops.append(avg_flops)
                en.append(energy)
                mem.append(mem_utilization)
                area.append(auc)
                params.append(num_parameters)
                files.append(file.name)
        return flops, en, mem, area, files, params
    elif model_type == cfg.model_types[3]:  # MobileNet
        flops, en, mem, area, params, files = [], [], [], [], [], []

        for file in model_dirs[3].glob("mobilenet_model_*.pth"):
            if "early_stopped" in file.stem:
                continue
            print(f"Testing MobileNet model: {file.name}")
            lr = float(file.stem.split("_")[-1])
            print(f"Model parameters - Learning Rate: {lr}")
            avg_flops, energy, mem_utilization, auc, lr, num_parameters = evaluate_mobilenet_model(
                MobileNet,
                file,
                lr,
                classes,
                test_data,
                device,
            )
            flops.append(avg_flops)
            en.append(energy)
            mem.append(mem_utilization)
            area.append(auc)
            params.append(num_parameters)
            files.append(file.name)
        return flops, en, mem, area, files, params
    elif model_type in _RESNET_MODEL_MAP:  # ResNet
        flops, en, mem, area, params, files = [], [], [], [], [], []
        resnet_dir = model_dirs[cfg.model_types.index(model_type)]

        for file in resnet_dir.glob("resnet_model_*.pth"):
            if "early_stopped" in file.stem:
                continue
            print(f"Testing ResNet model: {file.name}")
            lr = float(file.stem.split("_")[-1])
            print(f"Model parameters - Learning Rate: {lr}")
            avg_flops, energy, mem_utilization, auc, lr, num_parameters = evaluate_resnet_model(
                _RESNET_MODEL_MAP[model_type],
                file,
                lr,
                classes,
                test_data,
                device,
            )
            flops.append(avg_flops)
            en.append(energy)
            mem.append(mem_utilization)
            area.append(auc)
            params.append(num_parameters)
            files.append(file.name)
        return flops, en, mem, area, files, params
    elif model_type == "tensormera":  # TensorMERA
        flops, en, mem, area, params, files = [], [], [], [], [], []
        tensormera_dir = model_dirs[cfg.model_types.index(model_type)]

        for file in tensormera_dir.glob("tensormera_model_*.pth"):
            if "early_stopped" in file.stem:
                continue
            print(f"Testing TensorMERA model: {file.name}")
            num_layers, shrink_factor, lr = int(file.stem.split("_")[2]), float(file.stem.split("_")[3]), float(file.stem.split("_")[4])
            print(f"Model parameters - Num Layers: {num_layers}, Shrink Factor: {shrink_factor}, Learning Rate: {lr}")
            avg_flops, energy, mem_utilization, auc, num_parameters, _ = evaluate_tensormera_model(
                file,
                num_layers,
                shrink_factor,
                classes,
                test_data,
                device,
                image_size,
            )
            flops.append(avg_flops)
            en.append(energy)
            mem.append(mem_utilization)
            area.append(auc)
            params.append(num_parameters)
            files.append(file.name)
        return flops, en, mem, area, files, params

    raise ValueError(f"Unsupported model type: {model_type}")