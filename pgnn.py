from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from torch_geometric.nn import GCNConv, ChebConv, global_mean_pool
except ImportError as e:
    raise ImportError(
        "This version of pgnn.py requires torch_geometric.\n"
        "Install it first, for example:\n"
        "  pip install torch-geometric\n"
        "and the matching PyTorch extension packages for your environment."
    ) from e


# ============================================================
# Utilities
# ============================================================

def ensure_batch_vector(x: Tensor, batch: Optional[Tensor]) -> Tensor:
    """
    If batch is None, assume x belongs to a single graph.
    """
    if batch is None:
        return torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    return batch


class MLP(nn.Module):
    """
    Small MLP block used for projections / classifier heads.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        use_layernorm: bool = False,
    ):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ============================================================
# Input encoders
# ============================================================

class NodeFeatureEncoder(nn.Module):
    """
    Optional node-feature projection before graph convolutions.

    This is useful because:
    - x1 and x2 may have different feature dimensions
    - you may want to map them into model-friendly hidden widths
    """
    def __init__(
        self,
        x1_in_dim: int,
        x2_in_dim: int,
        x1_hidden_dim: int,
        x2_hidden_dim: int,
        dropout: float = 0.0,
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.x1_proj = MLP(
            in_dim=x1_in_dim,
            hidden_dim=max(x1_hidden_dim, x1_in_dim),
            out_dim=x1_hidden_dim,
            dropout=dropout,
            use_layernorm=use_layernorm,
        )
        self.x2_proj = MLP(
            in_dim=x2_in_dim,
            hidden_dim=max(x2_hidden_dim, x2_in_dim),
            out_dim=x2_hidden_dim,
            dropout=dropout,
            use_layernorm=use_layernorm,
        )

    def forward(self, x1: Tensor, x2: Tensor) -> tuple[Tensor, Tensor]:
        return self.x1_proj(x1), self.x2_proj(x2)


# ============================================================
# Sparse graph branches
# ============================================================

class SparseGCNBranch(nn.Module):
    """
    Local branch using sparse GCN message passing.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim, add_self_loops=False, normalize=True)
        self.conv2 = GCNConv(hidden_dim, out_dim, add_self_loops=False, normalize=True)

        self.norm1 = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
        self.norm2 = nn.LayerNorm(out_dim) if use_layernorm else nn.Identity()
        self.dropout = dropout

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class SparseChebBranch(nn.Module):
    """
    Global branch using sparse Chebyshev spectral convolution.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        K: int = 3,
        dropout: float = 0.0,
        use_layernorm: bool = False,
        normalization: str = "sym",
    ):
        super().__init__()
        self.conv1 = ChebConv(in_dim, hidden_dim, K=K, normalization=normalization)
        self.conv2 = ChebConv(hidden_dim, out_dim, K=K, normalization=normalization)

        self.norm1 = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
        self.norm2 = nn.LayerNorm(out_dim) if use_layernorm else nn.Identity()
        self.dropout = dropout

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# ============================================================
# Fusion heads
# ============================================================

class FusionHead(nn.Module):
    """
    Concatenation-based fusion.

    Node-level fusion:
        [f_local || f_global] -> linear/nonlinear projection

    Graph-level pooling:
        global_mean_pool
    """
    def __init__(
        self,
        in_dim_gcn: int,
        in_dim_cheb: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.0,
        use_layernorm: bool = False,
    ):
        super().__init__()
        fuse_in = in_dim_gcn + in_dim_cheb

        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        self.cls = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, f_local: Tensor, f_global: Tensor, batch: Optional[Tensor]) -> Tensor:
        batch = ensure_batch_vector(f_local, batch)
        fused = torch.cat([f_local, f_global], dim=-1)
        fused = self.fuse(fused)
        pooled = global_mean_pool(fused, batch)
        return self.cls(pooled)


class MultiplicativeFusionHead(nn.Module):
    """
    Multiplicative node-level fusion:
        proj(local) * proj(global)

    Then graph-level mean pooling and classification.
    """
    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.0,
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.proj_local = nn.Linear(feat_dim, feat_dim)
        self.proj_global = nn.Linear(feat_dim, feat_dim)

        self.cls = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, f_local: Tensor, f_global: Tensor, batch: Optional[Tensor]) -> Tensor:
        batch = ensure_batch_vector(f_local, batch)
        l = self.proj_local(f_local)
        g = self.proj_global(f_global)
        fused = l * g
        pooled = global_mean_pool(fused, batch)
        return self.cls(pooled)


# ============================================================
# Main PGNN model
# ============================================================

class PGNNNetSparse(nn.Module):
    """
    Sparse PGNN-style network for image-to-graph datasets.

    Expected inputs:
        x1:         (total_nodes, x1_in_dim)
        x2:         (total_nodes, x2_in_dim)
        edge_index: (2, total_edges)
        batch:      (total_nodes,) graph membership vector
        edge_weight: optional sparse edge weights

    This is compatible with the sparse_graph_collate_fn output.
    """
    def __init__(
        self,
        x1_in_dim: int,
        x2_in_dim: int,
        x1_hidden_dim: int,
        x2_hidden_dim: int,
        gcn_hidden: int,
        cheb_hidden: int,
        feat_dim: int,
        num_classes: int,
        cheb_k: int = 3,
        dropout: float = 0.0,
        use_layernorm: bool = False,
        fusion: str = "concat",
        cheb_normalization: str = "sym",
    ):
        super().__init__()

        if fusion not in ("concat", "multiply"):
            raise ValueError("fusion must be either 'concat' or 'multiply'.")

        self.encoder = NodeFeatureEncoder(
            x1_in_dim=x1_in_dim,
            x2_in_dim=x2_in_dim,
            x1_hidden_dim=x1_hidden_dim,
            x2_hidden_dim=x2_hidden_dim,
            dropout=dropout,
            use_layernorm=use_layernorm,
        )

        self.local_branch = SparseGCNBranch(
            in_dim=x1_hidden_dim,
            hidden_dim=gcn_hidden,
            out_dim=feat_dim,
            dropout=dropout,
            use_layernorm=use_layernorm,
        )

        self.global_branch = SparseChebBranch(
            in_dim=x2_hidden_dim,
            hidden_dim=cheb_hidden,
            out_dim=feat_dim,
            K=cheb_k,
            dropout=dropout,
            use_layernorm=use_layernorm,
            normalization=cheb_normalization,
        )

        if fusion == "concat":
            self.head = FusionHead(
                in_dim_gcn=feat_dim,
                in_dim_cheb=feat_dim,
                hidden_dim=feat_dim,
                num_classes=num_classes,
                dropout=dropout,
                use_layernorm=use_layernorm,
            )
        else:
            self.head = MultiplicativeFusionHead(
                feat_dim=feat_dim,
                hidden_dim=feat_dim,
                num_classes=num_classes,
                dropout=dropout,
                use_layernorm=use_layernorm,
            )

        self.fusion = fusion
        self.num_classes = num_classes

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        x1, x2 = self.encoder(x1, x2)

        f_local = self.local_branch(x1, edge_index, edge_weight=edge_weight)
        f_global = self.global_branch(x2, edge_index, edge_weight=edge_weight)

        logits = self.head(f_local, f_global, batch=batch)
        return logits


# ============================================================
# Optional convenience wrapper
# ============================================================

@dataclass
class PGNNConfig:
    x1_in_dim: int
    x2_in_dim: int
    x1_hidden_dim: int = 32
    x2_hidden_dim: int = 32
    gcn_hidden: int = 64
    cheb_hidden: int = 64
    feat_dim: int = 128
    num_classes: int = 10
    cheb_k: int = 3
    dropout: float = 0.1
    use_layernorm: bool = True
    fusion: str = "concat"
    cheb_normalization: str = "sym"


def build_pgnn_model(cfg: PGNNConfig) -> PGNNNetSparse:
    return PGNNNetSparse(
        x1_in_dim=cfg.x1_in_dim,
        x2_in_dim=cfg.x2_in_dim,
        x1_hidden_dim=cfg.x1_hidden_dim,
        x2_hidden_dim=cfg.x2_hidden_dim,
        gcn_hidden=cfg.gcn_hidden,
        cheb_hidden=cfg.cheb_hidden,
        feat_dim=cfg.feat_dim,
        num_classes=cfg.num_classes,
        cheb_k=cfg.cheb_k,
        dropout=cfg.dropout,
        use_layernorm=cfg.use_layernorm,
        fusion=cfg.fusion,
        cheb_normalization=cfg.cheb_normalization,
    )


# ============================================================
# Minimal smoke test
# ============================================================

if __name__ == "__main__":
    # Example with 2 tiny graphs batched together.
    # Graph 1: 4 nodes, Graph 2: 3 nodes
    total_nodes = 7
    x1_dim = 3
    x2_dim = 3

    x1 = torch.randn(total_nodes, x1_dim)
    x2 = torch.randn(total_nodes, x2_dim)

    # Example sparse edge list across the concatenated batch.
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 4, 5, 5, 6],
            [1, 0, 2, 1, 3, 2, 5, 4, 6, 5],
        ],
        dtype=torch.long,
    )

    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1], dtype=torch.long)

    model = PGNNNetSparse(
        x1_in_dim=x1_dim,
        x2_in_dim=x2_dim,
        x1_hidden_dim=16,
        x2_hidden_dim=16,
        gcn_hidden=32,
        cheb_hidden=32,
        feat_dim=64,
        num_classes=10,
        cheb_k=3,
        dropout=0.1,
        use_layernorm=True,
        fusion="concat",
    )

    logits = model(x1=x1, x2=x2, edge_index=edge_index, batch=batch)
    print("logits shape:", logits.shape)  # expected: (2, 10)