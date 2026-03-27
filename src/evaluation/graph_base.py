from collections import defaultdict
from typing import Iterable, Mapping, MutableMapping, Sequence, Set, Tuple

import networkx as nx
from networkx.algorithms import community

# 型定義
Edge = Tuple[str, str] | Tuple[str, str, float]
Partition = Sequence[Iterable[str]] | Mapping[str, str]
Attributes = Mapping[str, str | int]


def calculate_modularity(
    nodes: Iterable[str],
    edges: Iterable[Edge],
    partition: Partition | None = None,
) -> float:
    """ネットワークのモジュラリティ（コミュニティ分割の質）を計算する。"""
    graph = _build_graph(nodes, edges)
    if graph.number_of_edges() == 0:
        return 0.0

    communities = _prepare_partition(graph, partition)
    # 重み付きエッジがある場合は考慮する
    weight_key = "weight" if any("weight" in data for _, _, data in graph.edges(data=True)) else None

    return float(community.modularity(graph, communities, weight=weight_key))


def calculate_homophily(
    nodes: Iterable[str],
    edges: Iterable[Edge],
    node_attributes: Attributes,
    attribute_name: str = "category"
) -> float:
    """
    属性アソートタティビティ（ホモフィリー指標）を計算する。
    1.0に近いほど似た属性同士が繋がり、-1.0に近いほど異なる属性同士が繋がっている。
    """
    graph = _build_graph(nodes, edges)
    if graph.number_of_edges() == 0:
        return 0.0

    # ノードに属性を付与
    nx.set_node_attributes(graph, node_attributes, attribute_name)
    
    try:
        # 属性に基づく相関係数を計算
        return float(nx.attribute_assortativity_coefficient(graph, attribute_name))
    except (ValueError, ZeroDivisionError):
        return 0.0


def compute_basic_metrics(
    nodes: Iterable[str],
    edges: Iterable[Edge],
    partition: Partition | None = None,
    node_attributes: Attributes | None = None,
    attribute_name: str = "category"
) -> Mapping[str, float]:
    """グラフ全体の基本指標、モジュラリティ、およびホモフィリーを計算して返す。"""

    graph = _build_graph(nodes, edges)
    if graph.number_of_nodes() == 0:
        return {
            "density": 0.0,
            "average_clustering": 0.0,
            "transitivity": 0.0,
            "connected_components": 0.0,
            "largest_component_ratio": 0.0,
            "avg_shortest_path_length": 0.0,
            "modularity": 0.0,
            "homophily": 0.0,
        }

    # --- 基本指標の計算 ---
    density = nx.density(graph)
    avg_clustering = nx.average_clustering(graph, weight="weight")
    transitivity = nx.transitivity(graph)

    components = list(nx.connected_components(graph))
    component_count = float(len(components))
    largest_component_size = max(len(component) for component in components)
    largest_component_ratio = largest_component_size / graph.number_of_nodes()

    if largest_component_size <= 1:
        avg_shortest_path_length = 0.0
    else:
        largest_subgraph = graph.subgraph(max(components, key=len)).copy()
        avg_shortest_path_length = nx.average_shortest_path_length(largest_subgraph, weight="weight")

    # --- 追加指標（モジュラリティ & ホモフィリー） ---
    modularity = calculate_modularity(nodes, edges, partition)
    
    homophily = 0.0
    if node_attributes:
        homophily = calculate_homophily(nodes, edges, node_attributes, attribute_name)

    return {
        "density": float(density),
        "average_clustering": float(avg_clustering),
        "transitivity": float(transitivity),
        "connected_components": component_count,
        "largest_component_ratio": float(largest_component_ratio),
        "avg_shortest_path_length": float(avg_shortest_path_length),
        "modularity": modularity,
        "homophily": homophily,
    }


def _build_graph(nodes: Iterable[str], edges: Iterable[Edge]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(nodes)

    for edge in edges:
        if len(edge) == 3:
            source, target, weight = edge
            graph.add_edge(source, target, weight=float(weight))
        elif len(edge) == 2:
            source, target = edge  # type: ignore[misc]
            graph.add_edge(source, target)
        else:
            raise ValueError("Edges must be 2-tuples or 3-tuples")

    return graph


def _prepare_partition(graph: nx.Graph, partition: Partition | None) -> Sequence[Set[str]]:
    if partition is None:
        # パーティションがない場合は自動検出（greedy_modularity）
        detected = community.greedy_modularity_communities(graph, weight="weight")
        return [set(group) for group in detected]

    if isinstance(partition, Mapping):
        groups: MutableMapping[str, Set[str]] = defaultdict(set)
        for node, label in partition.items():
            if node in graph:
                groups[str(label)].add(node)
        return [members for members in groups.values() if members]

    normalized = [set(group) for group in partition]
    node_set = set(graph.nodes)
    return [group & node_set for group in normalized if group & node_set]


__all__ = ["calculate_modularity", "calculate_homophily", "compute_basic_metrics"]
