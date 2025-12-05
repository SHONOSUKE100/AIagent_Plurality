"""Command-line entry point to evaluate graph-level echo chamber metrics."""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Tuple

from graph_base import calculate_modularity, compute_basic_metrics


def load_like_graph(conn: sqlite3.Connection) -> Tuple[List[str], List[Tuple[str, str, float]]]:
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


def compute_metrics(sqlite_path: Path | str) -> None:
	conn = sqlite3.connect(sqlite_path)
	try:
		nodes, edges = load_like_graph(conn)
	finally:
		conn.close()

	metrics = compute_basic_metrics(nodes, edges)
	modularity_score = calculate_modularity(nodes, edges)

	print("Graph metrics (like-based user network):")
	for name, value in metrics.items():
		print(f"  {name}: {value:.4f}")
	print(f"  modularity: {modularity_score:.4f}")


def main(argv: Iterable[str] | None = None) -> None:
	parser = argparse.ArgumentParser(description="Compute graph metrics from the simulation database.")
	parser.add_argument(
		"--sqlite-path",
		default="data/twitter_simulation.db",
		help="Path to the simulation SQLite database.",
	)

	args = parser.parse_args(list(argv) if argv is not None else None)

	compute_metrics(Path(args.sqlite_path))


if __name__ == "__main__":
	main()