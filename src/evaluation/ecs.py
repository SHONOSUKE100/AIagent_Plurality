"""Echo Chamber Score (ECS) implementation based on arXiv:2307.04668."""

from __future__ import annotations

import json
import sqlite3
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
from networkx.algorithms import community

from .graph_data import load_like_graph_from_connection


def load_user_embeddings(conn: sqlite3.Connection) -> Dict[int, np.ndarray]:
    """Load user embeddings from the SQLite database."""

    try:
        rows = conn.execute("SELECT user_id, embedding FROM user_embedding").fetchall()
    except sqlite3.OperationalError:
        # Table missing: return empty so callers can gracefully skip ECS
        return {}
    embeddings: Dict[int, np.ndarray] = {}
    for row in rows:
        try:
            vec = np.array(json.loads(row["embedding"]), dtype=float)
            embeddings[int(row["user_id"])] = vec
        except Exception:
            continue
    return embeddings


def detect_communities_from_likes(
    nodes: Sequence[str],
    edges: Sequence[Tuple[str, str, float]],
) -> List[set[int]]:
    """Detect communities on the like graph using greedy modularity."""

    if not nodes or not edges:
        return []

    g = nx.Graph()
    g.add_nodes_from(int(n) for n in nodes)
    for u, v, w in edges:
        g.add_edge(int(u), int(v), weight=float(w))

    detected = community.greedy_modularity_communities(g)
    return [set(map(int, comm)) for comm in detected]


def _pairwise_distances(mat: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Compute pairwise Euclidean distances; optional max-normalization to [0,1]."""

    diffs = mat[:, None, :] - mat[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    if normalize:
        max_val = float(np.max(dists))
        if max_val > 0:
            dists = dists / max_val
    return dists


def compute_ecs(
    embeddings: Dict[int, np.ndarray],
    communities: Sequence[set[int]],
    normalize_distances: bool = True,
    ) -> Tuple[Dict[int, float], float, Dict[int, int]]:
    """Compute ECS*(omega) per community and ECS(Omega) overall.

    Args:
        embeddings: Mapping user_id -> embedding vector.
        communities: List of sets of user_ids representing communities.
        normalize_distances: Whether to normalize pairwise distances to [0,1].

    Returns:
        ecs_per_comm: {community_index: ECS*(omega)} keyed by the original index of ``communities``.
        ecs_global: ECS(Omega)
        valid_sizes: {community_index: number_of_users_with_embeddings_used_for_score}
    """

    if not embeddings or not communities:
        return {}, 0.0, {}

    user_ids = sorted(embeddings)
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    mat = np.vstack([embeddings[u] for u in user_ids])
    dist_matrix = _pairwise_distances(mat, normalize=normalize_distances)

    ecs_per_comm: Dict[int, float] = {}
    valid_sizes: Dict[int, int] = {}

    for orig_idx, comm in enumerate(communities):
        comm_valid = [u for u in comm if u in embeddings]
        if len(comm_valid) < 2:
            continue
        valid_sizes[orig_idx] = len(comm_valid)

        scores: list[float] = []
        comm_indices = np.array([user_to_idx[u] for u in comm_valid], dtype=int)

        for u in comm_valid:
            u_idx = user_to_idx[u]
            # Cohesion lambda_u
            others = comm_indices[comm_indices != u_idx]
            if len(others) == 0:
                lambda_u = 0.0
            else:
                # mean(dist) = sum / (|omega|-1); adjust to sum / |omega| to match paper
                mean_same = float(np.mean(dist_matrix[u_idx, others]))
                lambda_u = mean_same * (len(others) / len(comm_valid))

            # Separation delta_u
            min_avg = float("inf")
            for other_idx, other_comm in enumerate(communities):
                if other_idx == orig_idx:
                    continue
                other_valid = [v for v in other_comm if v in embeddings]
                if not other_valid:
                    continue
                other_indices = [user_to_idx[v] for v in other_valid]
                if other_indices:
                    avg_dist = float(np.mean(dist_matrix[u_idx, other_indices]))
                    min_avg = min(min_avg, avg_dist)
            delta_u = min_avg if min_avg != float("inf") else 0.0

            m = max(delta_u, lambda_u)
            s_u = 0.0 if m == 0 else (m + delta_u - lambda_u) / (2.0 * m)
            scores.append(s_u)

        ecs_star = float(np.mean(scores)) if scores else 0.0
        ecs_per_comm[orig_idx] = ecs_star

    ecs_global = float(np.mean(list(ecs_per_comm.values()))) if ecs_per_comm else 0.0
    return ecs_per_comm, ecs_global, valid_sizes


def compute_ecs_from_db(conn: sqlite3.Connection) -> Tuple[Dict[int, float], float, List[set[int]]]:
    """Convenience wrapper to compute ECS from a SQLite connection."""

    nodes, edges = load_like_graph_from_connection(conn)
    communities = detect_communities_from_likes(nodes, edges)
    embeddings = load_user_embeddings(conn)
    ecs_per_comm, ecs_global, valid_sizes = compute_ecs(embeddings, communities)
    return ecs_per_comm, ecs_global, communities, valid_sizes


__all__ = [
    "compute_ecs",
    "compute_ecs_from_db",
    "detect_communities_from_likes",
    "load_user_embeddings",
]
