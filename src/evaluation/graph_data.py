import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
NodeId = str
WeightedEdge = Tuple[str, str, float]
NodeAttributes = Dict[str, str] # ホモフィリー計算用にノードごとの属性を格納


def load_like_graph_from_connection(
    conn: sqlite3.Connection,
    attribute_column: str = "category"  # 属性として使用するカラム名
) -> Tuple[List[NodeId], List[WeightedEdge], NodeAttributes]:
    """
    ユーザーノード、重み付きエッジ、および属性データを取得する。
    """

    conn.row_factory = sqlite3.Row

    # 1. ノードと属性の取得
    # userテーブルから ID と 指定された属性カラムを取得します
    node_rows = conn.execute(f"SELECT user_id, {attribute_column} FROM user").fetchall()
    
    nodes = []
    node_attributes = {}
    
    for row in node_rows:
        u_id = str(row["user_id"])
        nodes.append(u_id)
        # 属性（例：部署、性格タイプ、エージェントの種類など）を辞書に格納
        node_attributes[u_id] = str(row[attribute_column])

    # 2. エッジ（いいね関係）の取得
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
        # 無向グラフとして扱うためにソート（必要に応じて有向グラフに変更可能）
        edge = tuple(sorted((src, dst)))
        weight_map[edge] += 1

    edges = [(u, v, float(weight)) for (u, v), weight in weight_map.items()]
    
    return nodes, edges, node_attributes


def load_like_graph(
    sqlite_path: Path | str, 
    attribute_column: str = "category"
) -> Tuple[List[NodeId], List[WeightedEdge], NodeAttributes]:
    """SQLiteファイルを開き、グラフ構造とノード属性を返す。"""

    conn = sqlite3.connect(sqlite_path)
    try:
        return load_like_graph_from_connection(conn, attribute_column)
    finally:
        conn.close()


__all__ = ["load_like_graph", "load_like_graph_from_connection"]
# graph_data.py 内

def load_like_graph_from_connection(
    conn: sqlite3.Connection,
    attribute_column: str = "role"  # ここを "category" から実際の列名（例: role）に変える
) -> Tuple[List[NodeId], List[WeightedEdge], NodeAttributes]:
    ...