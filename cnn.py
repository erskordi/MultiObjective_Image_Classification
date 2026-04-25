import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
from zeus.monitor import ZeusMonitor

class CNNImageClassifier(nn.Module):
    """A simple CNN architecture for image classification."""
    def __init__(self, 
                 num_classes=10,
                 input_channels=3, 
                 output_channels=32, 
                 kernel_size=3, 
                 padding=1,
                 num_conv_layers=2,
                 maxpool_kernel=2,
                 maxpool_stride=2):
        super(CNNImageClassifier, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.num_conv_layers = num_conv_layers
        self.maxpool_kernel = maxpool_kernel
        self.maxpool_stride = maxpool_stride
        self.use_cuda_amp = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler(enabled=self.use_cuda_amp)
        self.num_classes = num_classes

        self.conv_layers = nn.ModuleList()
        in_ch = self.input_channels
        for _ in range(self.num_conv_layers):
            self.conv_layers.append(
                nn.Conv2d(in_ch, self.output_channels, kernel_size=self.kernel_size, padding=self.padding)
            )
            in_ch = self.output_channels
        self.pool = nn.MaxPool2d(self.maxpool_kernel, self.maxpool_stride)

        # Global average pooling makes the classifier robust to different input sizes.
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(self.output_channels, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """Defines the forward pass of the CNN model."""
        for conv in self.conv_layers:
            x = F.relu(conv(x))
            x = self.pool(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))   # Apply first fully connected layer and ReLU activation
        x = self.fc2(x)           # Apply output layer for classification
        return x
    
    def train_model(self, dataloader, loss_fn, optimizer, device, monitor):
        """Train the CNN model for one epoch and measure energy and memory usage."""
        self.train() # Put the model in training mode.
        total_loss = 0
        monitor.begin_window("epoch")
        mem_before = torch.cuda.memory_allocated() if device == "cuda" else 0
        for X, y in dataloader:
            # Place the model and predictions on the same device.
            X = X.to(device, non_blocking=(device == "cuda"))
            y = y.to(device, non_blocking=(device == "cuda"))
            
            # Compute prediction error
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                pred = self(X)
                loss = loss_fn(pred, y)

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            total_loss += loss.item()
        measurement = monitor.end_window("epoch")
        mem_after = torch.cuda.memory_allocated() if device == "cuda" else 0
        mem_utilized = mem_after - mem_before
        return total_loss, round(measurement.total_energy, 2), round(mem_utilized / 1024**2, 2)
    
    @torch.no_grad()
    def test_model(self, dataloader, loss_fn, device):
        """Evaluate the model on the test dataset and return accuracy, average loss, and average FLOPs per sample."""
        monitor = ZeusMonitor(gpu_indices=[0])  # Create a monitor instance for testing
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.eval() # Put the model in testing mode.
        total_flops = 0
        test_loss, correct = 0, 0
        monitor.begin_window("testing")
        mem_before = torch.cuda.memory_allocated() if device == "cuda" else 0
        all_preds, all_labels = [], []

        for X, y in dataloader:
            with FlopCounterMode(self) as flop_counter:
                self(X.to(device))
            total_flops += flop_counter.get_total_flops()
            
            X = X.to(device, non_blocking=(device == "cuda"))
            y = y.to(device, non_blocking=(device == "cuda"))
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                pred = self(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        measurement = monitor.end_window("testing")
        mem_after = torch.cuda.memory_allocated() if device == "cuda" else 0
        mem_utilized = mem_after - mem_before
        test_loss /= num_batches
        correct /= size
        average_flops = total_flops / size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        return 100*correct, test_loss, average_flops, round(measurement.total_energy, 2), round(mem_utilized / 1024**2, 2), all_preds, all_labels
    
    def save_model(self, path):
        """Save the model's state dictionary to the specified path."""
        torch.save(self.state_dict(), path)