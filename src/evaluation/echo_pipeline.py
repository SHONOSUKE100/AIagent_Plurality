"""High-level pipeline to train EchoGAE on the like graph and compute ECS."""

from __future__ import annotations

from dataclasses import dataclass
import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import torch

from .echogae import EchoGAE, train_echo_gae
from .ecs import compute_ecs_detailed, detect_communities_on_graph, load_user_embeddings
from .graph_data import load_like_graph_from_connection


NodeId = int
Edge = Tuple[str, str, float]


@dataclass
class EchoGAEResult:
    user_ids: list[NodeId]
    embeddings: np.ndarray  # shape (N, embedding_dim)
    communities: list[set[NodeId]]
    ecs_per_comm: dict[int, float]
    ecs_global: float
    valid_sizes: dict[int, int]
    user_scores: dict[int, float]
    graph: nx.Graph
    model: EchoGAE


def _build_feature_matrix(
    user_ids: Sequence[NodeId],
    *,
    raw_embeddings: Dict[NodeId, np.ndarray] | None,
    content_embeddings: Dict[NodeId, np.ndarray] | None,
    feature_dim: int = 128,
    random_seed: int = 42,
) -> torch.Tensor:
    rng = np.random.default_rng(random_seed)
    dim_candidates: list[int] = []
    if raw_embeddings:
        dim_candidates.append(len(next(iter(raw_embeddings.values()))))
    if content_embeddings:
        dim_candidates.append(len(next(iter(content_embeddings.values()))))
    dim = dim_candidates[0] if dim_candidates else feature_dim

    mat = np.zeros((len(user_ids), dim), dtype=np.float32)
    for idx, uid in enumerate(user_ids):
        parts: list[np.ndarray] = []
        if raw_embeddings and uid in raw_embeddings:
            parts.append(np.asarray(raw_embeddings[uid], dtype=np.float32))
        if content_embeddings and uid in content_embeddings:
            parts.append(np.asarray(content_embeddings[uid], dtype=np.float32))

        if parts:
            normed: list[np.ndarray] = []
            for p in parts:
                if p.shape[0] < dim:
                    pad_width = dim - p.shape[0]
                    p = np.concatenate([p, np.zeros(pad_width, dtype=np.float32)], axis=0)
                elif p.shape[0] > dim:
                    p = p[:dim]
                normed.append(p)
            vec = np.mean(normed, axis=0)
        else:
            vec = rng.normal(loc=0.0, scale=1.0, size=dim)
        mat[idx] = vec
    return torch.from_numpy(mat)


def load_user_content_embeddings(conn: sqlite3.Connection) -> Dict[NodeId, np.ndarray]:
    """Aggregate post/comment embeddings per user (mean)."""

    conn.row_factory = sqlite3.Row
    user_to_vecs: dict[int, list[np.ndarray]] = {}

    def _add_rows(rows, user_key: str, emb_key: str):
        for row in rows:
            try:
                uid = int(row[user_key])
                vec = np.array(json.loads(row[emb_key]), dtype=np.float32)
                user_to_vecs.setdefault(uid, []).append(vec)
            except Exception:
                continue

    try:
        post_rows = conn.execute(
            """
            SELECT p.user_id AS user_id, pe.embedding AS embedding
            FROM post_embedding AS pe
            JOIN post AS p ON pe.post_id = p.post_id
            """
        ).fetchall()
        _add_rows(post_rows, "user_id", "embedding")
    except sqlite3.OperationalError:
        pass

    try:
        comment_rows = conn.execute(
            """
            SELECT c.user_id AS user_id, ce.embedding AS embedding
            FROM comment_embedding AS ce
            JOIN comment AS c ON ce.comment_id = c.comment_id
            """
        ).fetchall()
        _add_rows(comment_rows, "user_id", "embedding")
    except sqlite3.OperationalError:
        pass

    aggregated: Dict[int, np.ndarray] = {}
    for uid, vecs in user_to_vecs.items():
        aggregated[uid] = np.mean(vecs, axis=0)
    return aggregated


def _build_adjacency(
    user_ids: Sequence[NodeId],
    edges: Sequence[Edge],
    *,
    min_weight: float = 1.0,
) -> torch.Tensor:
    """Return symmetric sparse adjacency (binary)."""

    n = len(user_ids)
    id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    idx_rows: list[int] = []
    idx_cols: list[int] = []
    for u, v, w in edges:
        if float(w) < min_weight:
            continue
        ui = id_to_idx.get(int(u))
        vi = id_to_idx.get(int(v))
        if ui is None or vi is None:
            continue
        idx_rows.extend([ui, vi])
        idx_cols.extend([vi, ui])
    if not idx_rows:
        indices = torch.zeros((2, 0), dtype=torch.long)
        values = torch.zeros(0, dtype=torch.float32)
    else:
        indices = torch.tensor([idx_rows, idx_cols], dtype=torch.long)
        values = torch.ones(len(idx_rows), dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, (n, n))


def _build_graph(user_ids: Sequence[NodeId], edges: Sequence[Edge], *, min_weight: float = 1.0) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(user_ids)
    for u, v, w in edges:
        if float(w) < min_weight:
            continue
        g.add_edge(int(u), int(v), weight=float(w))
    return g


def run_echo_gae_pipeline(
    conn,
    *,
    hidden_dim: int = 128,
    embedding_dim: int = 64,
    feature_dim: int = 128,
    epochs: int = 100,
    lr: float = 1e-3,
    neg_ratio: float = 1.0,
    community_method: str = "louvain",
    min_edge_weight: float = 1.0,
    device: str | torch.device | None = None,
    checkpoint_path: str | Path | None = None,
    random_seed: int = 42,
    use_comm_size_denominator: bool = True,
) -> EchoGAEResult:
    """End-to-end EchoGAE + ECS for the like graph in the SQLite DB.

    Uses per-user embeddings if present and augments them with the mean of that
    user's post/comment embeddings to better reflect authored content.
    """

    nodes, edges = load_like_graph_from_connection(conn)
    user_ids = sorted({int(n) for n in nodes})
    if not user_ids or not edges:
        raise ValueError("Like graph is empty; cannot train EchoGAE.")

    user_embeddings = load_user_embeddings(conn)
    content_embeddings = load_user_content_embeddings(conn)
    features = _build_feature_matrix(
        user_ids,
        raw_embeddings=user_embeddings,
        content_embeddings=content_embeddings,
        feature_dim=feature_dim,
        random_seed=random_seed,
    )
    adj = _build_adjacency(user_ids, edges, min_weight=min_edge_weight)
    graph = _build_graph(user_ids, edges, min_weight=min_edge_weight)

    # Train EchoGAE
    model, z = train_echo_gae(
        adj,
        features,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        epochs=epochs,
        lr=lr,
        neg_ratio=neg_ratio,
        device=device,
        checkpoint_path=checkpoint_path,
    )

    communities = detect_communities_on_graph(graph, method=community_method)
    embedding_dict = {uid: z[i].numpy() for i, uid in enumerate(user_ids)}
    ecs_per_comm, ecs_global, valid_sizes, user_scores = compute_ecs_detailed(
        embedding_dict,
        communities,
        use_comm_size_denominator=use_comm_size_denominator,
    )

    return EchoGAEResult(
        user_ids=list(user_ids),
        embeddings=z.numpy(),
        communities=communities,
        ecs_per_comm=ecs_per_comm,
        ecs_global=ecs_global,
        valid_sizes=valid_sizes,
        user_scores=user_scores,
        graph=graph,
        model=model,
    )


__all__ = ["run_echo_gae_pipeline", "EchoGAEResult", "load_user_content_embeddings"]
