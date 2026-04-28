from xml.parsers.expat import model

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
from torchsummary import summary
from torch_flops import TorchFLOPsByFX
from zeus.monitor import ZeusMonitor

from help_functions import confusion_matrix, auroc

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=16):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, embed_dim,
                                kernel_size=patch_size, stride=patch_size)
    def forward(self, X):
        X = self.conv2d(X)  # shape [B=Batch, C=Channels, H=Height, W=Width]
        X = X.flatten(start_dim=2)  # shape [B, C, H * W]
        return X.transpose(1, 2)  # shape [B, H * W, C]

class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 num_classes=1000, embed_dim=768, depth=12, num_heads=12,
                 ff_dim=3072, dropout=0.1):
        super().__init__()

       
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        cls_init = torch.randn(1, 1, embed_dim) * 0.02
        self.cls_token = nn.Parameter(cls_init)  # shape [1, 1, E=embed_dim]
        num_patches = (img_size // patch_size) ** 2  # num_patches (noted L)
        pos_init = torch.randn(1, num_patches + 1, embed_dim) * 0.02
        self.pos_embed = nn.Parameter(pos_init)  # shape [1, 1 + L, E]
        self.dropout = nn.Dropout(p=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, num_classes)

    def forward(self, X):
        Z = self.patch_embed(X)  # shape [B, L, E]
        cls_expd = self.cls_token.expand(Z.shape[0], -1, -1)  # shape [B, 1, E]
        Z = torch.cat((cls_expd, Z), dim=1)  # shape [B, 1 + L, E]
        Z = Z + self.pos_embed
        Z = self.dropout(Z)
        Z = self.encoder(Z)  # shape [B, 1 + L, E]
        Z = self.layer_norm(Z[:, 0])  # shape [B, E]
        logits = self.output(Z) # shape [B, C]
        return logits

class MDN(nn.Module):
    def __init__(self, input_dim=768, embed_dim=768, hidden_size=64, k=2):
        super().__init__()
        
        self.linear1 = nn.Linear(input_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        
        self.mean = nn.Linear(hidden_size, k)
        self.std = nn.Linear(hidden_size, k)
        self.weights = nn.Linear(hidden_size, k)
        
    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        
        mean = self.mean(x)
        std = torch.exp(self.std(x))
        weights = F.softmax(self.weights(x), dim=1)
        
        return mean, std, weights
    
    def sample(self, x):
        mean, std, weights = self.forward(x)
        
        # Sample mixture components
        cat = torch.distributions.Categorical(weights)
        components = cat.sample()
        
        # Sample from selected component
        normals = torch.distributions.Normal(mean, std)
        samples = normals.sample()
        
        # Gather samples according to chosen component
        samples = samples.gather(1, components.unsqueeze(1))
        
        return samples
    
class ViTImageClassifier(ViT):
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 num_classes=10, embed_dim=768, depth=12, num_heads=12,
                 ff_dim=3072, dropout=0.1, use_mdn=False):
        super().__init__(img_size, patch_size, in_channels,
                         num_classes, embed_dim, depth, num_heads,
                         ff_dim, dropout)
        vit = ViT(img_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads, ff_dim, dropout)
        mdn = MDN(input_dim=embed_dim, embed_dim=embed_dim, hidden_size=64, k=15)
        self.patch_embed = vit.patch_embed
        self.cls_token = vit.cls_token
        self.pos_embed = vit.pos_embed
        self.dropout = vit.dropout
        self.encoder = vit.encoder
        self.layer_norm = vit.layer_norm
        self.output = nn.Linear(embed_dim, num_classes)
        self.use_mdn = use_mdn
        if use_mdn:
            self.mdn = mdn
        self.use_cuda_amp = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler(enabled=self.use_cuda_amp)

    def forward(self, X):
        Z = self.patch_embed(X)  # shape [B, L, E]
        cls_expd = self.cls_token.expand(Z.shape[0], -1, -1)  # shape [B, 1, E]
        Z = torch.cat((cls_expd, Z), dim=1)  # shape [B, 1 + L, E]
        Z = Z + self.pos_embed
        Z = self.dropout(Z)
        Z = self.encoder(Z)  # shape [B, 1 + L, E]
        Z = self.layer_norm(Z[:, 0])  # shape [B, E]
        logits = self.output(Z) # shape [B, C]
        if self.use_mdn:
            mean, std, weights = self.mdn(logits) # shape [B, C] -> (mean, std, weights)
            return logits, mean, std, weights
        return logits
    
    def train_model(self, dataloader, loss_fn, optimizer, device, monitor):
        """ Train the ViT model for one epoch and measure energy and memory usage."""
        size = len(dataloader.dataset)
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
                if self.use_mdn:
                    logits, _, _, _ = pred
                    loss = loss_fn(logits, y)
                else:
                    loss = loss_fn(pred, y)

            # Backpropagation
            # This step updates the parameter values inside your network.
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
        """Evaluate the model on the test dataset and measure energy and memory usage."""
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.eval() # Put the model in testing mode.
        total_flops = 0
        test_loss, correct = 0, 0
        monitor = ZeusMonitor(gpu_indices=[0])  # Create a monitor instance for testing ZeusMonitor(gpu_indices=[0])  # Create a monitor instance for testing
        monitor.begin_window("testing")
        torch.cuda.empty_cache()  # Clear GPU cache before testing
        mem_before = 0.0
        all_preds, all_labels = [], []

        for X, y in dataloader:
            X = X.to(device, non_blocking=(device == "cuda"))
            y = y.to(device, non_blocking=(device == "cuda"))
            with FlopCounterMode(display=False) as flop_counter:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                    pred = self(X)
                    if self.use_mdn:
                        logits, _, _, _ = pred
                        all_preds.append(logits.cpu())
                        all_labels.append(y.cpu())
                        test_loss += loss_fn(logits, y).item()
                        correct += (logits.argmax(1) == y).type(torch.float).sum().item()
                    else:
                        all_preds.append(pred.cpu())
                        all_labels.append(y.cpu())
                        test_loss += loss_fn(pred, y).item()
                        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            total_flops += flop_counter.get_total_flops()
        test_loss /= num_batches
        correct /= size
        measurement = monitor.end_window("testing")
        mem_after = torch.cuda.memory_allocated() if device == "cuda" else 0
        mem_utilized = mem_after - mem_before
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        # Calculate conf matrix and AUC for all batches
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)


        return 100*correct, test_loss, total_flops , measurement.total_energy, mem_utilized / 1024**2, all_preds, all_labels
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)