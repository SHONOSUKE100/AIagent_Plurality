"""Visualization helpers for EchoGAE embeddings, graph layout, and ECS."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.express as px
# ForceAtlas2 backends: prefer fa2l (pure Python), fallback to fa2 if present.
ForceAtlas2 = None
try:
    from fa2l import ForceAtlas2  # type: ignore
except Exception:
    try:  # pragma: no cover - optional dependency path
        from fa2 import ForceAtlas2  # type: ignore
    except Exception:
        ForceAtlas2 = None
from sklearn.manifold import TSNE

try:
    import umap
except Exception:  # pragma: no cover - optional
    umap = None


def _community_colors(communities: Sequence[set[int]]) -> Dict[int, str]:
    cmap = plt.get_cmap("tab20")
    lookup: dict[int, str] = {}
    for idx, comm in enumerate(communities):
        color = cmap(idx % cmap.N)
        for node in comm:
            lookup[int(node)] = color
    return lookup


def project_embeddings(z: np.ndarray, *, method: str = "tsne", random_state: int = 42) -> np.ndarray:
    """Reduce Z (N x d) to 2D using t-SNE or UMAP."""

    if z.shape[1] <= 2:
        return z[:, :2]
    method = method.lower()
    if method == "umap" and umap is not None:
        reducer = umap.UMAP(n_components=2, random_state=random_state, min_dist=0.1)
        return reducer.fit_transform(z)
    tsne = TSNE(n_components=2, random_state=random_state, init="pca", learning_rate="auto")
    return tsne.fit_transform(z)


def plot_embeddings_matplotlib(
    coords: np.ndarray,
    user_ids: Sequence[int],
    communities: Sequence[set[int]],
    *,
    title: str = "EchoGAE embeddings",
    user_scores: Dict[int, float] | None = None,
) -> plt.Figure:
    color_lookup = _community_colors(communities)
    colors = [color_lookup.get(int(uid), "#6b7280") for uid in user_ids]

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=35, edgecolors="#111827", linewidths=0.4)
    if user_scores:
        scores = [user_scores.get(int(uid), 0.0) for uid in user_ids]
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=scores,
            cmap="viridis",
            s=45,
            edgecolors="#111827",
            linewidths=0.4,
        )
        cbar = fig.colorbar(scatter, ax=ax, label="s(u)")
        cbar.ax.tick_params(labelsize=8)

    for i, uid in enumerate(user_ids):
        ax.text(coords[i, 0], coords[i, 1], str(uid), fontsize=7, alpha=0.8)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(alpha=0.2)
    return fig


def plot_embeddings_plotly(
    coords: np.ndarray,
    user_ids: Sequence[int],
    communities: Sequence[set[int]],
    *,
    user_scores: Dict[int, float] | None = None,
    title: str = "EchoGAE embeddings",
):
    labels = []
    for idx, comm in enumerate(communities):
        for uid in comm:
            labels.append((int(uid), idx))
    label_map = dict(labels)
    comm_labels = [label_map.get(int(uid), -1) for uid in user_ids]
    hover = [f"user {uid}" for uid in user_ids]
    color_arg = comm_labels
    color_continuous_scale = None
    if user_scores:
        color_arg = [user_scores.get(int(uid), 0.0) for uid in user_ids]
        color_continuous_scale = "Viridis"
    fig = px.scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        color=color_arg,
        hover_name=[str(u) for u in user_ids],
        title=title,
        color_continuous_scale=color_continuous_scale,
    )
    fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color="#111")), selector=dict(mode="markers"))
    fig.update_layout(showlegend=user_scores is None)
    return fig


def compute_forceatlas2_layout(graph: nx.Graph, iterations: int = 1000) -> Dict[int, Tuple[float, float]]:
    if ForceAtlas2 is None:
        # Fallback to spring layout if ForceAtlas2 backend unavailable
        pos = nx.spring_layout(graph, seed=42)
        return {int(k): (float(x), float(y)) for k, (x, y) in pos.items()}

    fa2 = ForceAtlas2(
        outboundAttractionDistribution=False,
        linLogMode=False,
        adjustSizes=False,
        edgeWeightInfluence=1.0,
        jitterTolerance=1.0,
        barnesHutOptimize=True,
        barnesHutTheta=1.2,
        scalingRatio=2.0,
        strongGravityMode=False,
        gravity=1.0,
        verbose=False,
    )
    pos = fa2.forceatlas2_networkx_layout(graph, iterations=iterations)
    return {int(k): tuple(map(float, v)) for k, v in pos.items()}


def plot_forceatlas2(graph: nx.Graph, communities: Sequence[set[int]], *, iterations: int = 1000) -> plt.Figure:
    pos = compute_forceatlas2_layout(graph, iterations=iterations)
    color_lookup = _community_colors(communities)
    fig, ax = plt.subplots(figsize=(7, 5))

    # edges first
    for u, v in graph.edges():
        x1, y1 = pos[int(u)]
        x2, y2 = pos[int(v)]
        ax.plot([x1, x2], [y1, y2], color="#cbd5e1", alpha=0.25, linewidth=0.6)

    xs, ys, colors = [], [], []
    for node in graph.nodes():
        xs.append(pos[int(node)][0])
        ys.append(pos[int(node)][1])
        colors.append(color_lookup.get(int(node), "#6b7280"))

    ax.scatter(xs, ys, c=colors, s=50, edgecolors="#0f172a", linewidths=0.5)
    ax.set_title("ForceAtlas2 layout (like graph)")
    ax.axis("off")
    return fig


def plot_ecs_bars(ecs_per_comm: Dict[int, float], ecs_global: float) -> plt.Figure:
    labels = [f"ω{idx+1}" for idx in sorted(ecs_per_comm)]
    values = [ecs_per_comm[idx] for idx in sorted(ecs_per_comm)]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(labels, values, color="#2563eb", alpha=0.9)
    ax.axhline(ecs_global, color="#ef4444", linestyle="--", linewidth=1.2, label=f"ECS(Ω)={ecs_global:.3f}")
    ax.set_ylim(0, 1)
    ax.set_ylabel("ECS*(ω)")
    ax.set_title("Community-level ECS*")
    ax.legend()
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    return fig


def plot_user_scores(coords: np.ndarray, user_ids: Sequence[int], scores: Dict[int, float]) -> plt.Figure:
    values = [scores.get(int(uid), 0.0) for uid in user_ids]
    fig, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap="viridis", s=40, edgecolors="#111827", linewidths=0.4)
    ax.set_title("User-level s(u)")
    ax.axis("off")
    cbar = fig.colorbar(scatter, ax=ax, label="s(u)")
    cbar.ax.tick_params(labelsize=8)
    return fig


__all__ = [
    "project_embeddings",
    "plot_embeddings_matplotlib",
    "plot_embeddings_plotly",
    "compute_forceatlas2_layout",
    "plot_forceatlas2",
    "plot_ecs_bars",
    "plot_user_scores",
]
