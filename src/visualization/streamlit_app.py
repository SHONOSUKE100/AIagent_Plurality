"""Streamlit dashboard for the AI agent simulation data stored in SQLite.

The app replaces the NeoDash setup and reads directly from the simulation
database to surface key metrics, posts with their comments, follow relations,
and like interactions between users and posts.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on the path when running via `streamlit run`
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.graph_base import calculate_modularity, compute_basic_metrics
from src.evaluation.graph_data import load_like_graph_from_connection


DEFAULT_DB_PATH = Path("data/twitter_simulation.db")
LATEST_RUN_FILE = Path("results/latest_run.txt")


@st.cache_resource(show_spinner=False)
def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
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


def build_follow_graph(conn: sqlite3.Connection, user_limit: int = 40) -> nx.DiGraph:
    users = load_top_users(conn, user_limit)
    if users.empty:
        return nx.DiGraph()

    user_ids = users["user_id"].tolist()
    placeholders = ",".join(["?"] * len(user_ids))
    edge_query = f"""
        SELECT follower_id, followee_id
        FROM follow
        WHERE follower_id IN ({placeholders})
          AND followee_id IN ({placeholders})
    """
    edges = fetch_df(conn, edge_query, params=user_ids + user_ids)

    graph = nx.DiGraph()
    for _, row in users.iterrows():
        graph.add_node(
            int(row["user_id"]),
            label=row["user_name"],
            followers=int(row["num_followers"]),
            followings=int(row["num_followings"]),
        )

    for _, row in edges.iterrows():
        graph.add_edge(int(row["follower_id"]), int(row["followee_id"]))

    return graph


def build_like_graph(conn: sqlite3.Connection, post_limit: int = 25, edge_limit: int = 400) -> nx.Graph:
    posts = fetch_df(
        conn,
        """
        SELECT post_id, content, user_id
        FROM post
        ORDER BY created_at DESC
        LIMIT ?
        """,
        params=[post_limit],
    )

    if posts.empty:
        return nx.Graph()

    post_ids = posts["post_id"].tolist()
    placeholders = ",".join(["?"] * len(post_ids))
    likes = fetch_df(
        conn,
        f"""
        SELECT l.user_id, l.post_id, u.user_name, p.content
        FROM like AS l
        JOIN user AS u ON u.user_id = l.user_id
        JOIN post AS p ON p.post_id = l.post_id
        WHERE l.post_id IN ({placeholders})
        LIMIT ?
        """,
        params=post_ids + [edge_limit],
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


def _spread_positions(n: int) -> List[float]:
    if n <= 1:
        return [0.0]
    step = 2 / (n - 1)
    return [-1 + i * step for i in range(n)]


def plot_network(graph: nx.Graph, title: str, color_map: dict[str, str] | None = None) -> go.Figure | None:
    if not graph.nodes:
        return None

    # Prefer precomputed positions for bipartite graphs
    if all("x" in data and "y" in data for _, data in graph.nodes(data=True)):
        pos = {node: (data["x"], data["y"]) for node, data in graph.nodes(data=True)}
    else:
        pos = nx.spring_layout(graph, seed=42, k=0.6)

    edge_x: List[float] = []
    edge_y: List[float] = []
    for src, dst in graph.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.8, color="#BBBBBB"),
        hoverinfo="none",
        mode="lines",
    )

    node_x: List[float] = []
    node_y: List[float] = []
    texts: List[str] = []
    hovers: List[str] = []
    colors: List[str] = []
    sizes: List[float] = []

    for node, data in graph.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        label = str(data.get("label", node))
        texts.append(label)
        hovers.append(label)
        kind = data.get("kind")
        colors.append((color_map or {}).get(kind, "#228be6"))
        sizes.append(12 + 2 * graph.degree[node])

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=texts,
        textposition="top center",
        hovertext=hovers,
        marker=dict(
            size=sizes,
            color=colors,
            line=dict(width=0.5, color="#333333"),
            opacity=0.85,
        ),
        hoverinfo="text",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=title,
        showlegend=False,
        hovermode="closest",
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#0f1116",
        plot_bgcolor="#0f1116",
        font=dict(color="#e5e7eb"),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
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
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("フォロー関係が見つかりませんでした。")


def render_like_graph(conn: sqlite3.Connection) -> None:
    st.subheader("ユーザーとコンテンツのいいね関係")
    graph = build_like_graph(conn)
    fig = plot_network(graph, "User-Post likes", color_map={"user": "#38bdf8", "post": "#f97316"})
    if fig:
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("いいね情報が見つかりませんでした。")


def render_top_users(conn: sqlite3.Connection) -> None:
    st.subheader("フォロワー数トップ")
    top_users = load_top_users(conn, 20)
    if top_users.empty:
        st.info("ユーザーデータが見つかりませんでした。")
        return
    st.dataframe(top_users, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="AI Agent Simulation Dashboard", layout="wide")
    st.title("AI Agent Simulation Dashboard (Streamlit)")
    st.caption("SQLiteのデータを直接読み込み、簡易的に状況を把握するダッシュボードです。")

    with st.sidebar:
        st.header("設定")
        default_path = _default_db_path()
        db_path = st.text_input("SQLiteのパス", value=str(default_path))
        st.caption(f"デフォルトは最新のrun (results/latest_run.txt) が指す {default_path}")

    db_path_resolved = Path(db_path)
    if not db_path_resolved.exists():
        st.error(f"データベースが見つかりません: {db_path_resolved}")
        return

    conn = _connect(str(db_path_resolved))

    render_summary(conn)

    render_metrics(conn)

    col1, col2 = st.columns([1, 1])
    with col1:
        render_posts(conn)
    with col2:
        render_top_users(conn)

    render_follow_graph(conn)
    render_like_graph(conn)


if __name__ == "__main__":
    main()
