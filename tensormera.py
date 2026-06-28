import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode
from zeus.monitor import ZeusMonitor


# ---- TensorMERA Layer with Constraints ----
class ConstrainedTensorMERALayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_sites):
        super().__init__()
        if num_sites % 2 != 0:
            raise ValueError("Number of sites must be even for pairing.")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_sites = num_sites
        self.num_pairs = num_sites // 2

        # Parameterize U indirectly to maintain unitarity
        self.U_params = nn.Parameter(torch.Tensor(in_dim, in_dim))
        # Initialize as skew-symmetric
        self.U_params.data = 0.01 * (torch.randn(in_dim, in_dim) -
                                      torch.randn(in_dim, in_dim).transpose(0, 1))

        # Isometry with orthogonality constraint
        self.W_params = nn.Parameter(torch.Tensor(out_dim, 2 * in_dim))
        # Properly initialize with a stable orthogonal matrix
        if out_dim <= 2 * in_dim:
            # Use proper orthogonal initialization when possible
            nn.init.orthogonal_(self.W_params)
        else:
            # If out_dim > 2*in_dim, can't use orthogonal directly
            temp = torch.zeros(out_dim, 2 * in_dim)
            nn.init.orthogonal_(temp[:2*in_dim, :])
            # Pad with small random values
            if out_dim > 2*in_dim:
                temp[2*in_dim:, :] = torch.randn(out_dim - 2*in_dim, 2 * in_dim) * 0.01
            self.W_params.data = temp

        # For bias
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def get_U(self):
        # Convert params to unitary matrix
        # Keep constrained matrix ops in fp32 for numerical stability.
        skew = (self.U_params - self.U_params.transpose(0, 1)).float()
        U = torch.matrix_exp(skew)
        if not torch.isfinite(U).all():
            U = torch.nan_to_num(U, nan=0.0, posinf=1.0, neginf=-1.0)
        return U.to(self.U_params.dtype)

    def get_W(self):
        """Ensure W has orthogonal rows (isometric property)"""
        # Clone to avoid modifying the original parameters
        W = self.W_params.clone().float()

        # Check for and handle NaN/Inf values
        if torch.isnan(W).any() or torch.isinf(W).any():
            # Reset problematic values to small random numbers
            mask = torch.isnan(W) | torch.isinf(W)
            W[mask] = torch.randn_like(W[mask]) * 0.01

        # Add a small regularization to improve numerical stability
        W = W + torch.eye(W.shape[0], W.shape[1], device=W.device, dtype=W.dtype) * 1e-6

        try:
            # Prefer linalg.svd for better numerical robustness.
            u, _, vh = torch.linalg.svd(W, full_matrices=False)
            W_orth = u @ vh
            if not torch.isfinite(W_orth).all():
                raise RuntimeError("Non-finite values in orthogonalized W")
            return W_orth.to(self.W_params.dtype)
        except RuntimeError:
            # If SVD fails, use a more stable but slower approach
            # Apply gradient steps toward orthogonality
            W_orth = W.clone()
            for _ in range(5):  # A few iterations toward orthogonality
                WtW = W_orth @ W_orth.transpose(-1, -2)
                I = torch.eye(WtW.shape[0], device=W.device, dtype=W_orth.dtype)
                # Step toward orthogonality
                grad = (WtW - I) @ W_orth
                W_orth = W_orth - 0.1 * grad

            # Normalize rows
            norms = torch.norm(W_orth, dim=1, keepdim=True)
            W_orth = W_orth / (norms + 1e-8)

            if not torch.isfinite(W_orth).all():
                W_orth = torch.nan_to_num(W_orth, nan=0.0, posinf=1.0, neginf=-1.0)

            return W_orth.to(self.W_params.dtype)

    def forward(self, x):
        _, sites, dim = x.shape
        if sites != self.num_sites or dim != self.in_dim:
            raise ValueError(f"Input mismatch: got ({sites}, {dim}), expected ({self.num_sites}, {self.in_dim})")

        # Get constrained parameters
        U = self.get_U()
        W = self.get_W()

        # Extract even and odd sites
        x_even = x[:, ::2, :]  # Shape: (b, p, d_in)
        x_odd = x[:, 1::2, :]  # Shape: (b, p, d_in)

        # Apply disentangler U to each site using einsum
        x_even_dis = torch.einsum('ik, bpk -> bpi', U, x_even)
        x_odd_dis = torch.einsum('ij, bpj -> bpi', U, x_odd)

        # Concatenate the disentangled pairs
        paired_dis = torch.cat([x_even_dis, x_odd_dis], dim=2)

        # Apply isometry W
        coarse_grained = torch.einsum('ij, bpj -> bpi', W, paired_dis) + self.bias

        return coarse_grained


# ---- Full TensorMERA Network ----
class ConstrainedTensorMERANet(nn.Module):
    def __init__(self, input_dim, input_sites, num_layers, shrink_factor=0.7):
        super().__init__()
        layers = []
        in_dim = input_dim
        sites = input_sites
        print(f"Is sites even? {sites % 2 == 0}")

        for _ in range(num_layers):
            out_dim = max(2, int(in_dim * shrink_factor))
            layer = ConstrainedTensorMERALayer(in_dim=in_dim, out_dim=out_dim, num_sites=sites)
            layers.append(layer)
            in_dim = out_dim
            sites = sites // 2

        self.layers = nn.ModuleList(layers)
        self.final_dim = in_dim
        self.final_sites = sites

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ---- MNIST Classifier with TensorMERA ----
class TensorMERA(nn.Module):
    """TensorMERA-based classifier with support for variable input channels and class counts."""
    
    def __init__(self, 
                 num_classes=10, 
                 input_channels=1,
                 input_dim=32, 
                 num_layers=3,
                 shrink_factor=0.7,
                 spatial_size=28):
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.input_dim = input_dim
        self.spatial_size = spatial_size
        # Keep AMP disabled: TensorMERA constrained ops (matrix_exp/SVD) are fp16-unstable.
        self.use_cuda_amp = False
        self.scaler = torch.amp.GradScaler(enabled=self.use_cuda_amp)

        # Project input channels to input_dim
        self.channel_proj = nn.Conv2d(input_channels, input_dim, kernel_size=1)

        # Create MERA model
        self.mera_model = ConstrainedTensorMERANet(
            input_dim=input_dim,
            input_sites=spatial_size * spatial_size,
            num_layers=num_layers,
            shrink_factor=shrink_factor
        )

        # Classifier
        self.classifier = nn.Linear(self.mera_model.final_dim * self.mera_model.final_sites, num_classes)

    def forward(self, x):
        # x shape: (batch_size, input_channels, spatial_size, spatial_size)
        batch_size = x.shape[0]

        # Project channels
        x = self.channel_proj(x)  # shape: (batch_size, input_dim, spatial_size, spatial_size)

        # Reshape for MERA
        x = x.permute(0, 2, 3, 1)  # shape: (batch_size, spatial_size, spatial_size, input_dim)
        x = x.reshape(batch_size, self.spatial_size * self.spatial_size, self.input_dim)  # flatten spatial

        # Apply MERA
        x = self.mera_model(x)

        # Classify
        x = x.reshape(batch_size, -1)
        return self.classifier(x)

    def train_model(self, dataloader, loss_fn, optimizer, device, monitor, lambda_reg=0.01):
        """Train for one epoch and return loss, energy, and memory used."""
        self.train()
        total_loss = 0.0
        monitor.begin_window("epoch")
        mem_before = torch.cuda.memory_allocated() if device == "cuda" else 0

        for X, y in dataloader:
            X = X.to(device, non_blocking=(device == "cuda"))
            y = y.to(device, non_blocking=(device == "cuda"))
            #print(f"Training batch with shape: {X.shape}, labels: {y.shape}")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda" and self.use_cuda_amp)):
                pred = self(X)
                if not torch.isfinite(pred).all():
                    pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
                criterion_loss = loss_fn(pred, y)
                
                # Add orthogonality regularization
                reg_loss = self._orthogonality_regularization(lambda_reg)
                if not torch.isfinite(reg_loss):
                    reg_loss = torch.zeros((), device=criterion_loss.device, dtype=criterion_loss.dtype)
                total_batch_loss = criterion_loss + reg_loss

            self.scaler.scale(total_batch_loss).backward()
            self.scaler.unscale_(optimizer)
            
            # Apply Riemannian optimization for constrained layers
            with torch.no_grad():
                for layer in self.mera_model.layers:
                    # Project U_params gradient to maintain skew-symmetry
                    if layer.U_params.grad is not None:
                        U_grad = layer.U_params.grad
                        skew_grad = 0.5 * (U_grad - U_grad.transpose(0, 1))
                        if not torch.isfinite(skew_grad).all():
                            skew_grad = torch.nan_to_num(skew_grad, nan=0.0, posinf=0.0, neginf=0.0)
                        layer.U_params.grad = skew_grad

                    # Project W_params gradient to maintain orthogonality
                    if layer.W_params.grad is not None:
                        W_grad = layer.W_params.grad
                        W = layer.W_params
                        W_grad_proj = W_grad - W @ W.transpose(0, 1) @ W_grad
                        if not torch.isfinite(W_grad_proj).all():
                            W_grad_proj = torch.nan_to_num(W_grad_proj, nan=0.0, posinf=0.0, neginf=0.0)
                        layer.W_params.grad = W_grad_proj

            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.scaler.step(optimizer)
            self.scaler.update()
            total_loss += criterion_loss.item()

        measurement = monitor.end_window("epoch")
        mem_after = torch.cuda.memory_allocated() if device == "cuda" else 0
        mem_utilized = mem_after - mem_before
        return total_loss, round(measurement.total_energy, 2), round(mem_utilized / 1024**2, 2)

    def _orthogonality_regularization(self, lambda_reg=0.01):
        """Add regularization to encourage orthogonality"""
        reg_loss = 0
        for layer in self.mera_model.layers:
            W = layer.W_params
            WWt = W @ W.transpose(0, 1)
            I = torch.eye(WWt.shape[0], device=W.device)
            reg_loss += torch.norm(WWt - I, p='fro')
        return lambda_reg * reg_loss

    @torch.no_grad()
    def test_model(self, dataloader, loss_fn, device):
        """Evaluate the model and return accuracy, loss, FLOPs, energy, memory, and predictions."""
        monitor = ZeusMonitor(gpu_indices=[0])
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.eval()
        total_flops = 0
        test_loss, correct = 0.0, 0.0
        monitor.begin_window("testing")
        mem_before = 0.0
        all_preds, all_labels = [], []

        for X, y in dataloader:
            X = X.to(device, non_blocking=(device == "cuda"))
            y = y.to(device, non_blocking=(device == "cuda"))
            with FlopCounterMode(display=False) as flop_counter:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda" and self.use_cuda_amp)):
                    pred = self(X)
                    if not torch.isfinite(pred).all():
                        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
                    test_loss += loss_fn(pred, y).item()
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                    all_preds.append(pred.cpu())
                    all_labels.append(y.cpu())
            total_flops += flop_counter.get_total_flops()

        measurement = monitor.end_window("testing")
        mem_after = torch.cuda.memory_allocated() if device == "cuda" else 0
        mem_utilized = mem_after - mem_before
        test_loss /= num_batches
        correct /= size
        average_flops = total_flops / size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of parameters: {num_parameters}")

        return (
            100 * correct,
            test_loss,
            average_flops,
            round(measurement.total_energy, 2),
            round(mem_utilized, 2),
            num_parameters,
            all_preds,
            all_labels,
        )

    def save_model(self, path):
        torch.save(self.state_dict(), path)
