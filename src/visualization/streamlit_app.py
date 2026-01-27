"""Streamlit dashboard for the AI agent simulation data stored in SQLite.

The app replaces the NeoDash setup and reads directly from the simulation
database to surface key metrics, posts with their comments, follow relations,
and like interactions between users and posts.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd
import streamlit as st
from networkx.algorithms.community import greedy_modularity_communities

# Ensure project root is on the path when running via `streamlit run`
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.graph_base import calculate_modularity, compute_basic_metrics
from src.evaluation.graph_data import load_like_graph_from_connection
from src.evaluation.ecs import compute_ecs_from_db
from src.evaluation.echo_pipeline import EchoGAEResult, run_echo_gae_pipeline
from src.visualization.echo_visualization import (
    plot_ecs_bars,
    plot_embeddings_plotly,
    plot_forceatlas2,
    plot_user_scores,
    project_embeddings,
)
import matplotlib_fontja


DEFAULT_DB_PATH = Path("data/twitter_simulation.db")
LATEST_RUN_FILE = Path("results/latest_run.txt")
STEP_METRICS_FILE = "step_metrics.csv"
STEP_SNAPSHOT_DIR = "step_snapshots"


@st.cache_resource(show_spinner=False)
def _connect(db_path: str) -> sqlite3.Connection:
    # Allow use across Streamlit threads; dashboard is read-only.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _default_db_path() -> Path:
    if LATEST_RUN_FILE.exists():
        try:
            latest_dir = Path(LATEST_RUN_FILE.read_text().strip())
            candidate = latest_dir / "simulation.db"
            if candidate.exists():
                return candidate
        except OSError:
            pass
    return DEFAULT_DB_PATH


def fetch_df(conn: sqlite3.Connection, query: str, params: Iterable | None = None) -> pd.DataFrame:
    return pd.read_sql_query(query, conn, params=params or [])


def fetch_scalar(conn: sqlite3.Connection, query: str) -> int:
    cur = conn.execute(query)
    row = cur.fetchone()
    return int(row[0]) if row else 0


def get_counts(conn: sqlite3.Connection) -> dict:
    return {
        "users": fetch_scalar(conn, "SELECT COUNT(*) FROM user"),
        "posts": fetch_scalar(conn, "SELECT COUNT(*) FROM post"),
        "comments": fetch_scalar(conn, "SELECT COUNT(*) FROM comment"),
        "likes": fetch_scalar(conn, "SELECT COUNT(*) FROM like"),
        "follows": fetch_scalar(conn, "SELECT COUNT(*) FROM follow"),
    }


def load_recent_posts(conn: sqlite3.Connection, limit: int = 30) -> pd.DataFrame:
    query = """
        SELECT p.post_id,
               u.user_name AS author,
               p.content,
               p.created_at,
               p.num_likes,
               p.num_dislikes,
               COUNT(c.comment_id) AS comment_count
        FROM post AS p
        JOIN user AS u ON u.user_id = p.user_id
        LEFT JOIN comment AS c ON c.post_id = p.post_id
        GROUP BY p.post_id
        ORDER BY p.created_at DESC
        LIMIT ?
    """
    return fetch_df(conn, query, params=[limit])


def load_comments_for_post(conn: sqlite3.Connection, post_id: int, limit: int = 50) -> pd.DataFrame:
    query = """
        SELECT c.comment_id,
               u.user_name AS commenter,
               c.content,
               c.created_at,
               c.num_likes,
               c.num_dislikes
        FROM comment AS c
        JOIN user AS u ON u.user_id = c.user_id
        WHERE c.post_id = ?
        ORDER BY c.created_at DESC
        LIMIT ?
    """
    return fetch_df(conn, query, params=[post_id, limit])


def load_top_users(conn: sqlite3.Connection, limit: int = 15) -> pd.DataFrame:
    query = """
        SELECT user_id,
               user_name,
               name,
               num_followers,
               num_followings
        FROM user
        ORDER BY num_followers DESC
        LIMIT ?
    """
    return fetch_df(conn, query, params=[limit])


def build_follow_graph(conn: sqlite3.Connection) -> nx.DiGraph:
    """Build a follow graph with all users."""

    users = fetch_df(
        conn,
        """
        SELECT user_id, user_name, num_followers, num_followings
        FROM user
        """,
    )
    if users.empty:
        return nx.DiGraph()

    edges = fetch_df(
        conn,
        """
        SELECT follower_id, followee_id
        FROM follow
        """,
    )

    graph = nx.DiGraph()
    for _, row in users.iterrows():
        graph.add_node(
            int(row["user_id"]),
            label=row["user_name"],
            kind="user",
            followers=int(row["num_followers"]),
            followings=int(row["num_followings"]),
        )

    for _, row in edges.iterrows():
        graph.add_edge(int(row["follower_id"]), int(row["followee_id"]))

    return graph


def build_like_graph(
    conn: sqlite3.Connection,
    post_limit: int | None = 25,
    edge_limit: int | None = 400,
) -> nx.Graph:
    limit_clause = "" if post_limit is None else "LIMIT ?"
    post_params: list[int] = [] if post_limit is None else [post_limit]
    posts = fetch_df(
        conn,
        f"""
        SELECT post_id, content, user_id
        FROM post
        ORDER BY created_at DESC
        {limit_clause}
        """,
        params=post_params,
    )

    if posts.empty:
        return nx.Graph()

    post_ids = posts["post_id"].tolist()
    placeholders = ",".join(["?"] * len(post_ids))
    like_limit_clause = "" if edge_limit is None else "LIMIT ?"
    like_params = post_ids + ([] if edge_limit is None else [edge_limit])
    likes = fetch_df(
        conn,
        f"""
        SELECT l.user_id, l.post_id, u.user_name, p.content
        FROM like AS l
        JOIN user AS u ON u.user_id = l.user_id
        JOIN post AS p ON p.post_id = l.post_id
        WHERE l.post_id IN ({placeholders})
        {like_limit_clause}
        """,
        params=like_params,
    )

    graph = nx.Graph()
    # Put users on level 0 and posts on level 1 for a simple bipartite layout
    user_ids = sorted(set(likes["user_id"].tolist())) if not likes.empty else []
    x_positions = _spread_positions(len(user_ids))
    for idx, user_id in enumerate(user_ids):
        user_name = likes.loc[likes["user_id"] == user_id, "user_name"].iloc[0]
        graph.add_node(int(user_id), label=user_name, kind="user", x=x_positions[idx], y=0)

    post_positions = _spread_positions(len(posts))
    for idx, (_, row) in enumerate(posts.iterrows()):
        label = row["content"][:40] + ("..." if len(row["content"]) > 40 else "")
        graph.add_node(int(row["post_id"]), label=label, kind="post", x=post_positions[idx], y=1)

    for _, row in likes.iterrows():
        graph.add_edge(int(row["user_id"]), int(row["post_id"]))

    return graph


def build_engagement_graph(
    conn: sqlite3.Connection,
    post_limit: int | None = 15,
    like_limit: int | None = 400,
    comment_limit: int | None = 150,
) -> nx.Graph:
    """Build a user-only interaction graph aggregated from posts/comments/likes."""

    post_limit_clause = "" if post_limit is None else "LIMIT ?"
    post_params: list[int] = [] if post_limit is None else [post_limit]
    posts = fetch_df(
        conn,
        f"""
        SELECT post_id, user_id AS author_id, content
        FROM post
        ORDER BY created_at DESC
        {post_limit_clause}
        """,
        params=post_params,
    )
    if posts.empty:
        return nx.Graph()

    post_ids = posts["post_id"].tolist()
    placeholders = ",".join(["?"] * len(post_ids))

    comment_limit_clause = "" if comment_limit is None else "LIMIT ?"
    comment_params = post_ids + ([] if comment_limit is None else [comment_limit])
    comments = fetch_df(
        conn,
        f"""
        SELECT comment_id, post_id, user_id AS author_id, content
        FROM comment
        WHERE post_id IN ({placeholders})
        ORDER BY created_at DESC
        {comment_limit_clause}
        """,
        params=comment_params,
    )

    like_limit_clause = "" if like_limit is None else "LIMIT ?"
    like_params = post_ids + ([] if like_limit is None else [like_limit])
    likes = fetch_df(
        conn,
        f"""
        SELECT user_id, post_id
        FROM like
        WHERE post_id IN ({placeholders})
        {like_limit_clause}
        """,
        params=like_params,
    )

    comment_likes = pd.DataFrame()
    if not comments.empty:
        comment_ids = comments["comment_id"].tolist()
        placeholders_comments = ",".join(["?"] * len(comment_ids))
        comment_like_limit_clause = "" if like_limit is None else "LIMIT ?"
        comment_like_params = comment_ids + ([] if like_limit is None else [like_limit])
        comment_likes = fetch_df(
            conn,
            f"""
            SELECT user_id, comment_id
            FROM comment_like
            WHERE comment_id IN ({placeholders_comments})
            {comment_like_limit_clause}
            """,
            params=comment_like_params,
        )

    # Aggregate interactions into user-user edges
    edge_weights: dict[tuple[int, int], int] = {}
    def _add_edge(u: int, v: int, increment: int = 1) -> None:
        if u == v:
            return
        key = tuple(sorted((u, v)))
        edge_weights[key] = edge_weights.get(key, 0) + increment

    for _, row in posts.iterrows():
        author = int(row["author_id"])
        # ensure node exists
        edge_weights.setdefault((author, author), 0)

    for _, row in comments.iterrows():
        comment_author = int(row["author_id"])
        post_author = int(row["post_id"])  # placeholder to fetch later
    # map post_id -> author for quick lookup
    post_author_map = {int(r["post_id"]): int(r["author_id"]) for _, r in posts.iterrows()}
    # comments interactions
    for _, row in comments.iterrows():
        comment_author = int(row["author_id"])
        post_author = post_author_map.get(int(row["post_id"]))
        if post_author is not None:
            _add_edge(comment_author, post_author)

    for _, row in likes.iterrows():
        liker = int(row["user_id"])
        post_author = post_author_map.get(int(row["post_id"]))
        if post_author is not None:
            _add_edge(liker, post_author)

    if not comment_likes.empty:
        comment_author_map = {int(r["comment_id"]): int(r["author_id"]) for _, r in comments.iterrows()}
        for _, row in comment_likes.iterrows():
            liker = int(row["user_id"])
            comment_author = comment_author_map.get(int(row["comment_id"]))
            if comment_author is not None:
                _add_edge(liker, comment_author)

    graph = nx.Graph()
    for (u, v), w in edge_weights.items():
        graph.add_node(u, kind="user")
        graph.add_node(v, kind="user")
        if u != v:
            graph.add_edge(u, v, weight=w)

    return graph


def _spread_positions(n: int) -> List[float]:
    if n <= 1:
        return [0.0]
    step = 2 / (n - 1)
    return [-1 + i * step for i in range(n)]


def plot_network(
    graph: nx.Graph,
    title: str,
    color_map: dict[str, str] | None = None,
    node_color_override: dict[int, str] | None = None,
    legend_handles: list[mpatches.Patch] | None = None,
):
    """Draw the network using matplotlib (bipartite layout if provided)."""

    if not graph.nodes:
        return None

    # Prefer precomputed positions for bipartite graphs
    if all("x" in data and "y" in data for _, data in graph.nodes(data=True)):
        pos = {node: (data["x"], data["y"]) for node, data in graph.nodes(data=True)}
    else:
        pos = nx.spring_layout(graph, seed=42, k=0.6)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(title, fontsize=12)
    ax.axis("off")

    # Edges
    nx.draw_networkx_edges(
        graph,
        pos,
        alpha=0.35,
        width=1.0,
        edge_color="#9ca3af",
        ax=ax,
    )

    # Nodes: color by kind if provided, else default
    default_color = "#2563eb"
    node_colors = []
    for node, data in graph.nodes(data=True):
        if node_color_override and node in node_color_override:
            node_colors.append(node_color_override[node])
            continue
        kind = data.get("kind")
        node_colors.append((color_map or {}).get(kind, default_color))

    node_sizes = [180 + 20 * graph.degree[n] for n in graph.nodes()]
    # Draw nodes only (no labels to avoid clutter)
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        linewidths=0.5,
        edgecolors="#111827",
        alpha=0.9,
        ax=ax,
    )

    combined_legend = list(legend_handles or [])
    if color_map:
        kinds_in_graph = {data.get("kind") for _, data in graph.nodes(data=True)}
        for kind, color in color_map.items():
            if kind in kinds_in_graph:
                label = str(kind) if kind is not None else "node"
                combined_legend.append(mpatches.Patch(color=color, label=label))
    if combined_legend:
        ax.legend(handles=combined_legend, loc="upper right", frameon=False, fontsize=8)

    fig.tight_layout()
    return fig


def compute_graph_metrics(conn: sqlite3.Connection) -> dict:
    nodes, edges = load_like_graph_from_connection(conn)
    if not nodes or not edges:
        return {}

    metrics = compute_basic_metrics(nodes, edges)
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    metrics["modularity"] = round(calculate_modularity(nodes, edges), 4)
    return metrics


def render_summary(conn: sqlite3.Connection) -> None:
    st.subheader("概要指標")
    counts = get_counts(conn)
    cols = st.columns(5)
    labels = [
        ("ユーザー", counts["users"]),
        ("ポスト", counts["posts"]),
        ("コメント", counts["comments"]),
        ("いいね", counts["likes"]),
        ("フォロー", counts["follows"]),
    ]
    for col, (label, value) in zip(cols, labels):
        col.metric(label=label, value=f"{value:,}")


def render_metrics(conn: sqlite3.Connection) -> None:
    st.subheader("評価指標 (いいね関係ネットワーク)")
    metrics = compute_graph_metrics(conn)
    if not metrics:
        st.info("指標を計算できる十分なデータがありません。")
        return

    # Display key metrics as metrics + table for detail
    metric_keys = [
        ("modularity", "モジュラリティ"),
        ("density", "密度"),
        ("average_clustering", "平均クラスタ係数"),
        ("transitivity", "トランジティビティ"),
    ]
    cols = st.columns(len(metric_keys))
    for col, (key, label) in zip(cols, metric_keys):
        col.metric(label, metrics.get(key, 0))

    st.write("詳細")
    st.dataframe(pd.DataFrame([metrics]), hide_index=True)


def render_ecs(conn: sqlite3.Connection) -> None:
    st.subheader("Echo Chamber Score (ECS)")
    ecs_per_comm, ecs_global, communities, valid_sizes = compute_ecs_from_db(conn)
    if not communities or not ecs_per_comm:
        st.info("ECSを計算するための埋め込みまたはコミュニティ情報が不足しています。")
        return

    st.metric("ECS(Ω)", f"{ecs_global:.4f}")

    rows = []
    for idx in sorted(ecs_per_comm):
        label = f"ω{idx+1}"
        size_valid = valid_sizes.get(idx, len(communities[idx]) if idx < len(communities) else 0)
        ecs_star = ecs_per_comm[idx]
        rows.append({"community": label, "size_used": size_valid, "ECS*": ecs_star})

    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True)

    # Matplotlib可視化: コミュニティサイズとECS*の関係を散布図で
    if not df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(df["size_used"], df["ECS*"], color="#2563eb")
        for _, row in df.iterrows():
            ax.text(row["size_used"], row["ECS*"] + 0.01, row["community"], fontsize=8)
        ax.set_xlabel("コミュニティ有効サイズ (埋め込みがあるユーザー数)")
        ax.set_ylabel("ECS*")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.set_title("コミュニティ別 ECS* とサイズ")
    st.pyplot(fig, use_container_width=True)

    st.bar_chart(df.set_index("community")[["ECS*"]], height=220)


@st.cache_resource(show_spinner=True)
def _cache_echo_gae(
    db_path: str,
    epochs: int,
    neg_ratio: float,
    community_method: str,
    min_edge_weight: float,
) -> EchoGAEResult:
    """Cache EchoGAE training so UI interactions stay fast."""

    conn = _connect(db_path)
    return run_echo_gae_pipeline(
        conn,
        epochs=epochs,
        neg_ratio=neg_ratio,
        community_method=community_method,
        min_edge_weight=min_edge_weight,
    )


def render_echo_gae_dashboard(db_path: str) -> None:
    st.subheader("EchoGAE + ECS（論文準拠）")
    cols = st.columns(4)
    with cols[0]:
        epochs = st.number_input("EchoGAE epochs", min_value=10, max_value=400, value=50, step=10)
    with cols[1]:
        neg_ratio = st.slider("Negative sampling ratio", min_value=0.5, max_value=5.0, value=1.0, step=0.5)
    with cols[2]:
        community_method = st.selectbox("コミュニティ検出", ["louvain", "leiden", "greedy"], index=0)
    with cols[3]:
        min_edge_weight = st.number_input("いいね閾値（重み）", min_value=1.0, max_value=10.0, value=1.0, step=1.0)

    projection_method = st.radio("埋め込みの次元圧縮", ["umap", "tsne"], horizontal=True, index=0)
    color_mode = st.radio("色付け", ["コミュニティ", "s(u)"], horizontal=True, index=0)

    try:
        result = _cache_echo_gae(str(db_path), int(epochs), float(neg_ratio), community_method, float(min_edge_weight))
    except ValueError as exc:
        st.info(str(exc))
        return
    except Exception as exc:
        st.error(f"EchoGAEの計算に失敗しました: {exc}")
        return

    coords = project_embeddings(result.embeddings, method=projection_method)
    st.metric("ECS(Ω)", f"{result.ecs_global:.4f}")

    tab1, tab2, tab3 = st.tabs(["Embedding", "ForceAtlas2", "ECS スコア"])

    with tab1:
        fig = plot_embeddings_plotly(
            coords,
            result.user_ids,
            result.communities,
            user_scores=result.user_scores if color_mode == "s(u)" else None,
            title=f"EchoGAE embeddings ({projection_method})",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fa_fig = plot_forceatlas2(result.graph, result.communities, iterations=800)
        st.pyplot(fa_fig, use_container_width=True)

    with tab3:
        ecs_fig = plot_ecs_bars(result.ecs_per_comm, result.ecs_global)
        st.pyplot(ecs_fig, use_container_width=True)

        scatter_fig = plot_user_scores(coords, result.user_ids, result.user_scores)
        st.pyplot(scatter_fig, use_container_width=True)


def render_ecs_embedding_graph(conn: sqlite3.Connection) -> None:
    """埋め込みとコミュニティに基づくユーザーグラフを可視化（ECS計算と同じ前処理）。"""
    ecs_per_comm, ecs_global, communities, valid_sizes = compute_ecs_from_db(conn)
    if not communities or not ecs_per_comm:
        return

    # Reuse embeddings and likes filtered to users with embeddings
    from src.evaluation.ecs import load_user_embeddings, detect_communities_from_likes

    embeddings = load_user_embeddings(conn)
    nodes, edges = load_like_graph_from_connection(conn)
    # Filter to embedded users
    embedded_users = set(embeddings)
    if not embedded_users:
        return

    min_weight_threshold = 2
    edges_embedded = [
        (int(u), int(v), w)
        for u, v, w in edges
        if int(u) in embedded_users and int(v) in embedded_users and w >= min_weight_threshold
    ]

    nodes_embedded = [n for n in nodes if int(n) in embedded_users]
    if not nodes_embedded or not edges_embedded:
        return

    # Recompute communities on embedded subgraph for visualization consistency
    comms_vis = detect_communities_from_likes(nodes_embedded, edges_embedded)
    user_to_comm = {}
    for idx, comm in enumerate(comms_vis):
        for u in comm:
            user_to_comm[int(u)] = idx

    # Positions from embeddings (first 2 dims); fallback to spring layout
    sample_vec = next(iter(embeddings.values()))
    use_embed_pos = len(sample_vec) >= 2
    if use_embed_pos:
        pos = {u: (embeddings[u][0], embeddings[u][1]) for u in embedded_users}
    else:
        g = nx.Graph()
        g.add_nodes_from(embedded_users)
        g.add_weighted_edges_from(edges_embedded)
        pos = nx.spring_layout(g, seed=42)

    cmap = plt.get_cmap("tab20")
    colors = []
    for u in embedded_users:
        comm_idx = user_to_comm.get(u, 0)
        colors.append(cmap(comm_idx % cmap.N))

    fig, ax = plt.subplots(figsize=(7, 5))
    # Draw edges lightly
    for u, v, w in edges_embedded:
        x1, y1 = pos[int(u)]
        x2, y2 = pos[int(v)]
        ax.plot([x1, x2], [y1, y2], color="#cbd5e1", alpha=0.3, linewidth=0.8)

    xs = [pos[u][0] for u in embedded_users]
    ys = [pos[u][1] for u in embedded_users]
    ax.scatter(xs, ys, c=colors, s=40, edgecolors="#111827", linewidths=0.4)

    ax.set_title("ユーザー埋め込み空間におけるコミュニティ（ECS算出と同前処理）")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)


def render_posts(conn: sqlite3.Connection) -> None:
    st.subheader("ポストとコメント")
    posts = load_recent_posts(conn)
    if posts.empty:
        st.info("ポストが見つかりませんでした。")
        return

    st.dataframe(posts[["author", "content", "created_at", "num_likes", "num_dislikes", "comment_count"]], hide_index=True)

    option_labels = [f"{row.post_id}: {row.content[:40]}" for row in posts.itertuples()]
    selected = st.selectbox("コメントを確認したいポスト", option_labels)
    selected_id = int(selected.split(":", 1)[0])
    comments = load_comments_for_post(conn, selected_id)
    st.write("コメント一覧")
    st.dataframe(comments, hide_index=True)


def render_follow_graph(conn: sqlite3.Connection) -> None:
    st.subheader("フォロー関係グラフ (上位ユーザー中心)")
    graph = build_follow_graph(conn)
    fig = plot_network(graph, "Follow network", color_map={None: "#22c55e", "user": "#22c55e"})
    if fig:
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("フォロー関係が見つかりませんでした。")


def render_like_graph(conn: sqlite3.Connection, show_all: bool = False) -> None:
    st.subheader("ユーザーとコンテンツのいいね関係")
    graph = build_like_graph(
        conn,
        post_limit=None if show_all else 25,
        edge_limit=None if show_all else 400,
    )
    fig = plot_network(graph, "User-Post likes", color_map={"user": "#38bdf8", "post": "#f97316"})
    if fig:
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("いいね情報が見つかりませんでした。")


def render_engagement_graph(conn: sqlite3.Connection, show_all: bool = False) -> None:
    st.subheader("ユーザー間インタラクション（ポスト/コメント経由を集約）")
    graph = build_engagement_graph(
        conn,
        post_limit=None if show_all else 15,
        like_limit=None if show_all else 400,
        comment_limit=None if show_all else 150,
    )
    fig = plot_network(
        graph,
        "Engagement network",
        color_map={"user": "#38bdf8"},
    )
    if fig:
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("十分なデータがありません。")


def render_echo_chamber_graph(conn: sqlite3.Connection) -> None:
    st.subheader("エコーチェンバー可視化（いいねネットワークのクラスター）")
    nodes, edges = load_like_graph_from_connection(conn)
    if not nodes or not edges:
        st.info("十分なデータがありません。")
        return

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_weighted_edges_from(edges)

    communities = list(greedy_modularity_communities(g))
    if not communities:
        st.info("クラスターを検出できませんでした。")
        return

    # Assign community colors
    cmap = plt.get_cmap("tab20")
    color_lookup: dict[int, str] = {}
    legend_handles: list[mpatches.Patch] = []
    for idx, comm in enumerate(communities):
        color = cmap(idx % cmap.N)
        for node in comm:
            color_lookup[int(node)] = color
        legend_handles.append(mpatches.Patch(color=color, label=f"cluster {idx+1}"))

    fig = plot_network(
        g,
        "Like network clusters",
        color_map={"user": "#38bdf8"},
        node_color_override=color_lookup,
        legend_handles=legend_handles,
    )
    if fig:
        st.pyplot(fig, use_container_width=True)


def render_top_users(conn: sqlite3.Connection) -> None:
    st.subheader("フォロワー数トップ")
    top_users = load_top_users(conn, 20)
    if top_users.empty:
        st.info("ユーザーデータが見つかりませんでした。")
        return
    st.dataframe(top_users, hide_index=True)


def _infer_run_dir(db_path: Path) -> Optional[Path]:
    """Best-effort guess of the run directory from a DB path."""

    if db_path.name == "simulation.db":
        return db_path.parent
    if db_path.parent.name == STEP_SNAPSHOT_DIR:
        return db_path.parent.parent
    # Fallback: assume parent is the run dir
    return db_path.parent if db_path.parent.exists() else None


def _list_snapshots(run_dir: Path) -> list[tuple[str, Path]]:
    """Return available snapshot labels and paths."""

    snap_dir = run_dir / STEP_SNAPSHOT_DIR
    if not snap_dir.exists():
        return []

    items: list[tuple[str, Path]] = []
    for path in sorted(snap_dir.glob("*.db")):
        items.append((path.stem, path))
    return items


def _load_step_metrics(run_dir: Path) -> pd.DataFrame:
    metrics_path = run_dir / STEP_METRICS_FILE
    if not metrics_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(metrics_path)
    # Ensure ordering by round/label
    if "round" in df.columns:
        df = df.sort_values(by=["round", "label"], na_position="last")
    return df


def render_step_metrics(metrics: pd.DataFrame) -> None:
    st.subheader("ステップ別メトリクス")
    if metrics.empty:
        st.info("step_metrics.csv が見つからないか、データがありません。")
        return
    df = metrics.copy()
    # Normalize types
    for col in df.columns:
        if col not in {"label"}:
            df[col] = pd.to_numeric(df[col], errors="ignore")

    st.dataframe(df, hide_index=True, use_container_width=True)

    # Cumulative view over steps
    count_cols = [c for c in ["users", "posts", "comments", "likes", "follows"] if c in df.columns]
    metric_cols = [c for c in ["modularity", "density", "average_clustering", "transitivity"] if c in df.columns]

    if count_cols:
        st.markdown("**累積カウントの推移**")
        st.line_chart(df.set_index("label")[count_cols], height=260)
        # 差分も表示（増減を把握する用）
        delta_df = df[["label"] + count_cols].copy()
        for col in count_cols:
            delta_df[f"Δ{col}"] = delta_df[col].diff()
        st.dataframe(delta_df[["label"] + [f"Δ{c}" for c in count_cols]], hide_index=True)

    if metric_cols:
        st.markdown("**ネットワーク指標の推移**")
        st.line_chart(df.set_index("label")[metric_cols], height=260)


def main() -> None:
    st.set_page_config(page_title="AI Agent Simulation Dashboard", layout="wide")
    st.title("AI Agent Simulation Dashboard (Streamlit)")
    st.caption("SQLiteのデータを直接読み込み、簡易的に状況を把握するダッシュボードです。")

    with st.sidebar:
        st.header("設定")
        default_path = _default_db_path()
        db_path = st.text_input("SQLiteのパス", value=str(default_path))
        st.caption(f"デフォルトは最新のrun (results/latest_run.txt) が指す {default_path}")
        base_run_dir = _infer_run_dir(Path(db_path))

        snapshot_choice = None
        show_all_graphs = st.checkbox("スナップショット全体を可視化（制限なし）", value=False)
        if base_run_dir and base_run_dir.exists():
            snapshots = _list_snapshots(base_run_dir)
            if snapshots:
                st.markdown("---")
                st.write("ステップスナップショット")
                labels = ["latest (simulation.db)"] + [label for label, _ in snapshots]
                selection = st.selectbox("表示するステップ", labels)
                if selection != "latest (simulation.db)":
                    snapshot_choice = dict(snapshots).get(selection)
                    if snapshot_choice:
                        db_path = str(snapshot_choice)
                metrics_df = _load_step_metrics(base_run_dir)
                render_step_metrics(metrics_df)

    db_path_resolved = Path(db_path)
    if not db_path_resolved.exists():
        st.error(f"データベースが見つかりません: {db_path_resolved}")
        return

    conn = _connect(str(db_path_resolved))

    render_summary(conn)

    render_metrics(conn)
    render_ecs(conn)
    render_echo_gae_dashboard(str(db_path_resolved))

    col1, col2 = st.columns([1, 1])
    with col1:
        render_posts(conn)
    with col2:
        render_top_users(conn)

    render_follow_graph(conn)
    render_like_graph(conn, show_all=show_all_graphs)
    render_engagement_graph(conn, show_all=show_all_graphs)
    render_echo_chamber_graph(conn)
    render_ecs_embedding_graph(conn)


if __name__ == "__main__":
    main()
