"""EchoGAE (Graph AutoEncoder) implementation for echo chamber detection.

This module follows the architecture described in arXiv:2307.04668:
- Encoder: 2-layer GCN -> Z in R^{N x 64}
- Decoder: Inner product with sigmoid
- Loss: Binary cross-entropy on A vs Â with negative sampling

The code supports both dense and sparse adjacency matrices and includes
checkpoint save/load helpers for reproducible training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

Tensor = torch.Tensor


def _to_tensor(x: Tensor | Sequence, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, dtype=dtype, device=device)


def _to_sparse(adj: Tensor) -> Tensor:
    """Ensure adjacency is a COO sparse tensor."""
    if adj.is_sparse:
        return adj.coalesce()
    idx = adj.nonzero(as_tuple=False).t()
    if idx.numel() == 0:
        # empty graph
        return torch.sparse_coo_tensor(idx, torch.zeros(0, device=adj.device), adj.shape, device=adj.device)
    values = adj[idx[0], idx[1]]
    return torch.sparse_coo_tensor(idx, values, adj.shape, device=adj.device)


def _normalize_sparse_adj(adj: Tensor, add_self_loops: bool = True, eps: float = 1e-8) -> Tensor:
    """Symmetric normalization D^{-1/2} A D^{-1/2} for sparse COO."""

    if add_self_loops:
        eye_idx = torch.arange(adj.size(0), device=adj.device)
        eye = torch.sparse_coo_tensor(
            torch.stack([eye_idx, eye_idx], dim=0),
            torch.ones_like(eye_idx, dtype=adj.dtype),
            adj.shape,
            device=adj.device,
        )
        adj = (adj + eye).coalesce()
    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv_sqrt = torch.pow(deg + eps, -0.5)
    row, col = adj.indices()
    values = adj.values() * deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return torch.sparse_coo_tensor(torch.stack([row, col], dim=0), values, adj.shape, device=adj.device)


def _normalize_dense_adj(adj: Tensor, add_self_loops: bool = True, eps: float = 1e-8) -> Tensor:
    if add_self_loops:
        adj = adj + torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
    deg = adj.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg + eps, -0.5)
    d_left = deg_inv_sqrt.unsqueeze(1)
    d_right = deg_inv_sqrt.unsqueeze(0)
    return adj * d_left * d_right


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation: Optional[nn.Module] = None, dropout: float = 0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: Tensor, adj_norm: Tensor) -> Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        support = x @ self.weight
        if adj_norm.is_sparse:
            out = torch.sparse.mm(adj_norm, support)
        else:
            out = adj_norm @ support
        out = out + self.bias
        if self.activation is not None:
            out = self.activation(out)
        return out


class GCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.gc1 = GraphConvolution(in_dim, hidden_dim, activation=nn.ReLU(), dropout=dropout)
        self.gc2 = GraphConvolution(hidden_dim, out_dim, activation=None, dropout=dropout)

    def forward(self, x: Tensor, adj_norm: Tensor) -> Tensor:
        h = self.gc1(x, adj_norm)
        z = self.gc2(h, adj_norm)
        return z


class EchoGAE(nn.Module):
    """Graph AutoEncoder with inner product decoder."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.encoder = GCNEncoder(in_dim, hidden_dim, out_dim, dropout=dropout)

    def forward(self, x: Tensor, adj_norm: Tensor) -> Tensor:
        return self.encoder(x, adj_norm)

    def decode_logits(self, z: Tensor) -> Tensor:
        return z @ z.t()

    def reconstruct(self, z: Tensor) -> Tensor:
        return torch.sigmoid(self.decode_logits(z))


@dataclass
class EchoGAECheckpoint:
    model_state: dict
    optimizer_state: Optional[dict]
    epoch: int
    metadata: dict


def _extract_edges(adj: Tensor) -> Tensor:
    """Return unique undirected edge indices (E x 2) from adjacency."""
    if adj.is_sparse:
        idx = adj.coalesce().indices()
        # keep upper triangle to avoid duplicates
        mask = idx[0] <= idx[1]
        return idx[:, mask].t()
    else:
        idx = torch.nonzero(adj > 0, as_tuple=False)
        mask = idx[:, 0] <= idx[:, 1]
        return idx[mask]


def _sample_negative_edges(num_nodes: int, num_samples: int, exclude: set[tuple[int, int]] | None = None, device: torch.device | None = None) -> Tensor:
    """Uniform negative edge sampler avoiding duplicates and self-loops."""
    device = device or torch.device("cpu")
    sampled = set()
    if exclude is None:
        exclude = set()
    attempts = 0
    max_attempts = max(num_samples * 5, 1000)
    while len(sampled) < num_samples and attempts < max_attempts:
        i = torch.randint(0, num_nodes, (1,), device=device).item()
        j = torch.randint(0, num_nodes, (1,), device=device).item()
        if i == j:
            attempts += 1
            continue
        a, b = (i, j) if i <= j else (j, i)
        if (a, b) in exclude or (a, b) in sampled:
            attempts += 1
            continue
        sampled.add((a, b))
    if len(sampled) < num_samples:
        # fallback: exhaustive scan
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if len(sampled) >= num_samples:
                    break
                if (i, j) in exclude or (i, j) in sampled:
                    continue
                sampled.add((i, j))
            if len(sampled) >= num_samples:
                break
    if not sampled:
        return torch.empty(0, 2, dtype=torch.long, device=device)
    arr = torch.tensor(sorted(sampled), dtype=torch.long, device=device)
    return arr


def reconstruction_loss(z: Tensor, pos_edges: Tensor, neg_edges: Tensor) -> Tensor:
    """BCE loss on sampled edges using inner-product logits."""

    def edge_logits(edges: Tensor) -> Tensor:
        if edges.numel() == 0:
            return torch.empty(0, device=z.device)
        src, dst = edges[:, 0], edges[:, 1]
        return torch.sum(z[src] * z[dst], dim=1)

    pos_logits = edge_logits(pos_edges)
    neg_logits = edge_logits(neg_edges)

    logits = torch.cat([pos_logits, neg_logits], dim=0)
    labels = torch.cat(
        [torch.ones_like(pos_logits, device=z.device), torch.zeros_like(neg_logits, device=z.device)],
        dim=0,
    )
    return nn.functional.binary_cross_entropy_with_logits(logits, labels)


def train_echo_gae(
    adj: Tensor | Sequence,
    features: Tensor | Sequence,
    *,
    hidden_dim: int = 128,
    embedding_dim: int = 64,
    epochs: int = 100,
    lr: float = 1e-3,
    neg_ratio: float = 1.0,
    weight_decay: float = 1e-4,
    device: str | torch.device | None = None,
    checkpoint_path: str | Path | None = None,
    verbose: bool = False,
) -> tuple[EchoGAE, Tensor]:
    """Train EchoGAE and return (model, node_embeddings).

    Args:
        adj: adjacency matrix (dense or sparse).
        features: input features X.
        neg_ratio: number of negative samples per positive edge.
        checkpoint_path: if provided, save checkpoint at the end.
    """

    device_obj = torch.device(device) if device is not None else torch.device("cpu")
    adj_t = _to_tensor(adj, device_obj)
    feat_t = _to_tensor(features, device_obj)

    # Normalized adjacency
    if adj_t.is_sparse:
        adj_norm = _normalize_sparse_adj(_to_sparse(adj_t))
    else:
        adj_norm = _normalize_dense_adj(adj_t)

    model = EchoGAE(feat_t.size(1), hidden_dim=hidden_dim, out_dim=embedding_dim)
    model.to(device_obj)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    pos_edges = _extract_edges(adj_t)
    # store as list for negative sampler
    pos_set = {(int(a), int(b)) for a, b in pos_edges.tolist()}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model(feat_t, adj_norm)

        num_neg = max(1, int(math.ceil(len(pos_set) * neg_ratio)))
        neg_edges = _sample_negative_edges(adj_t.size(0), num_neg, exclude=pos_set, device=device_obj)
        loss = reconstruction_loss(z, pos_edges, neg_edges)
        loss.backward()
        optimizer.step()

        if verbose and (epoch % 10 == 0 or epoch == epochs):
            print(f"[EchoGAE] epoch {epoch}/{epochs} loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        final_z = model(feat_t, adj_norm).detach().cpu()

    if checkpoint_path:
        ckpt = EchoGAECheckpoint(
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            epoch=epochs,
            metadata={"hidden_dim": hidden_dim, "embedding_dim": embedding_dim},
        )
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt.__dict__, path)

    return model, final_z


def load_checkpoint(path: str | Path, in_dim: int, hidden_dim: int = 128, embedding_dim: int = 64) -> EchoGAE:
    data = torch.load(path, map_location="cpu")
    model = EchoGAE(in_dim, hidden_dim=hidden_dim, out_dim=embedding_dim)
    model.load_state_dict(data["model_state"])
    return model


def encode_with_model(model: EchoGAE, features: Tensor | Sequence, adj: Tensor | Sequence, device: str | torch.device | None = None) -> Tensor:
    """Compute embeddings Z using a trained model."""

    device_obj = torch.device(device) if device is not None else torch.device("cpu")
    feat_t = _to_tensor(features, device_obj)
    adj_t = _to_tensor(adj, device_obj)
    adj_norm = _normalize_sparse_adj(_to_sparse(adj_t)) if adj_t.is_sparse else _normalize_dense_adj(adj_t)
    model = model.to(device_obj)
    model.eval()
    with torch.no_grad():
        z = model(feat_t, adj_norm)
    return z.detach().cpu()


__all__ = [
    "EchoGAE",
    "train_echo_gae",
    "encode_with_model",
    "load_checkpoint",
    "reconstruction_loss",
    "EchoGAECheckpoint",
]
