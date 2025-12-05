"""Helpers for loading graph structures from SQLite simulation data."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Tuple

NodeId = str
WeightedEdge = Tuple[str, str, float]


def load_like_graph_from_connection(
    conn: sqlite3.Connection,
) -> Tuple[List[NodeId], List[WeightedEdge]]:
    """Return user nodes and weighted edges based on post likes."""

    conn.row_factory = sqlite3.Row

    node_rows = conn.execute("SELECT user_id FROM user").fetchall()
    nodes = [str(row["user_id"]) for row in node_rows]

    like_rows = conn.execute(
        """
        SELECT l.user_id AS liker_id, p.user_id AS author_id
        FROM like AS l
        JOIN post AS p ON l.post_id = p.post_id
        WHERE l.user_id != p.user_id
        """
    ).fetchall()

    weight_map = defaultdict(int)
    for row in like_rows:
        src = str(row["liker_id"])
        dst = str(row["author_id"])
        if src == dst:
            continue
        edge = tuple(sorted((src, dst)))
        weight_map[edge] += 1

    edges = [(u, v, float(weight)) for (u, v), weight in weight_map.items()]
    return nodes, edges


def load_like_graph(sqlite_path: Path | str) -> Tuple[List[NodeId], List[WeightedEdge]]:
    """Open ``sqlite_path`` and return the like-based user graph."""

    conn = sqlite3.connect(sqlite_path)
    try:
        return load_like_graph_from_connection(conn)
    finally:
        conn.close()


__all__ = ["load_like_graph", "load_like_graph_from_connection"]
