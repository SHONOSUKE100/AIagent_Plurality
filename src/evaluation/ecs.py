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
    *,
    method: str = "louvain",
) -> List[set[int]]:
    """Detect communities on the like graph using Leiden/Louvain if available."""

    if not nodes or not edges:
        return []

    g = nx.Graph()
    g.add_nodes_from(int(n) for n in nodes)
    for u, v, w in edges:
        g.add_edge(int(u), int(v), weight=float(w))

    return detect_communities_on_graph(g, method=method)


def detect_communities_on_graph(graph: nx.Graph, *, method: str = "louvain") -> List[set[int]]:
    """Community detection wrapper preferring Leiden/Louvain, fallback to greedy modularity.

    Args:
        graph: undirected weighted user graph.
        method: "leiden" | "louvain" | "greedy".
    """

    method = (method or "louvain").lower()

    if method == "leiden":
        try:
            import igraph as ig  # type: ignore
            import leidenalg  # type: ignore
        except Exception:
            method = "louvain"
        else:
            # Convert to igraph for Leiden
            mapping = {node: idx for idx, node in enumerate(graph.nodes())}
            edges = [(mapping[u], mapping[v]) for u, v in graph.edges()]
            weights = [graph.edges[u, v].get("weight", 1.0) for u, v in graph.edges()]
            ig_graph = ig.Graph(list(mapping.values()), edges=edges, directed=False)
            ig_graph.es["weight"] = weights
            partition = leidenalg.find_partition(ig_graph, leidenalg.RBConfigurationVertexPartition, weights=weights)
            communities: list[set[int]] = [set() for _ in range(len(partition))]
            inv_mapping = {idx: node for node, idx in mapping.items()}
            for comm_idx, members in enumerate(partition):
                for m in members:
                    communities[comm_idx].add(int(inv_mapping[m]))
            return communities

    if method == "louvain":
        try:
            import community as community_louvain  # type: ignore
        except Exception:
            method = "greedy"
        else:
            partition = community_louvain.best_partition(graph, weight="weight")
            groups: dict[int, set[int]] = {}
            for node, label in partition.items():
                groups.setdefault(int(label), set()).add(int(node))
            return list(groups.values())

    detected = community.greedy_modularity_communities(graph, weight="weight")
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
    *,
    use_comm_size_denominator: bool = True,
    return_user_scores: bool = False,
) -> Tuple[Dict[int, float], float, Dict[int, int]] | Tuple[Dict[int, float], float, Dict[int, int], Dict[int, float]]:
    """Compute ECS*(omega) per community and ECS(Omega) overall.

    Args:
        embeddings: Mapping user_id -> embedding vector.
        communities: List of sets of user_ids representing communities.
        normalize_distances: Whether to normalize pairwise distances to [0,1].
        use_comm_size_denominator: If True, divide cohesion by |ω| per paper
            (otherwise fallback to |ω|-1).
        return_user_scores: When True, also return user-level s(u).

    Returns:
        ecs_per_comm: {community_index: ECS*(omega)} keyed by the original index of ``communities``.
        ecs_global: ECS(Omega)
        valid_sizes: {community_index: number_of_users_with_embeddings_used_for_score}
        user_scores (optional): {user_id: s(u)}
    """

    if not embeddings or not communities:
        if return_user_scores:
            return {}, 0.0, {}, {}
        return {}, 0.0, {}

    user_ids = sorted(embeddings)
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    mat = np.vstack([embeddings[u] for u in user_ids])
    dist_matrix = _pairwise_distances(mat, normalize=normalize_distances)

    ecs_per_comm: Dict[int, float] = {}
    valid_sizes: Dict[int, int] = {}
    user_scores: Dict[int, float] = {}

    for orig_idx, comm in enumerate(communities):
        comm_valid = [u for u in comm if u in embeddings]
        if len(comm_valid) < 2:
            continue
        valid_sizes[orig_idx] = len(comm_valid)

        scores: list[float] = []
        comm_indices = np.array([user_to_idx[u] for u in comm_valid], dtype=int)
        denom = len(comm_valid) if use_comm_size_denominator else max(1, len(comm_valid) - 1)

        for u in comm_valid:
            u_idx = user_to_idx[u]
            # Cohesion lambda_u
            others = comm_indices[comm_indices != u_idx]
            if len(others) == 0:
                lambda_u = 0.0
            else:
                # mean(dist) = sum / (|omega|-1); adjust to sum / |omega| to match paper
                mean_same = float(np.mean(dist_matrix[u_idx, others]))
                lambda_u = mean_same * (len(others) / denom)

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
            user_scores[u] = s_u

        ecs_star = float(np.mean(scores)) if scores else 0.0
        ecs_per_comm[orig_idx] = ecs_star

    ecs_global = float(np.mean(list(ecs_per_comm.values()))) if ecs_per_comm else 0.0
    if return_user_scores:
        return ecs_per_comm, ecs_global, valid_sizes, user_scores
    return ecs_per_comm, ecs_global, valid_sizes


def compute_ecs_from_db(
    conn: sqlite3.Connection,
    *,
    community_method: str = "louvain",
    embeddings_override: Dict[int, np.ndarray] | None = None,
    use_comm_size_denominator: bool = True,
) -> Tuple[Dict[int, float], float, List[set[int]], Dict[int, int]]:
    """Convenience wrapper to compute ECS from a SQLite connection."""

    nodes, edges = load_like_graph_from_connection(conn)
    communities = detect_communities_from_likes(nodes, edges, method=community_method)
    embeddings = embeddings_override if embeddings_override is not None else load_user_embeddings(conn)
    ecs_per_comm, ecs_global, valid_sizes = compute_ecs(
        embeddings,
        communities,
        use_comm_size_denominator=use_comm_size_denominator,
    )
    return ecs_per_comm, ecs_global, communities, valid_sizes


def compute_ecs_detailed(
    embeddings: Dict[int, np.ndarray],
    communities: Sequence[set[int]],
    *,
    normalize_distances: bool = True,
    use_comm_size_denominator: bool = True,
) -> Tuple[Dict[int, float], float, Dict[int, int], Dict[int, float]]:
    """Return ECS with user-level scores included."""

    ecs_per_comm, ecs_global, valid_sizes, user_scores = compute_ecs(
        embeddings,
        communities,
        normalize_distances=normalize_distances,
        use_comm_size_denominator=use_comm_size_denominator,
        return_user_scores=True,
    )
    return ecs_per_comm, ecs_global, valid_sizes, user_scores


__all__ = [
    "compute_ecs",
    "compute_ecs_from_db",
    "compute_ecs_detailed",
    "detect_communities_from_likes",
    "detect_communities_on_graph",
    "load_user_embeddings",
]
