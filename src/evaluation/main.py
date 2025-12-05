"""Command-line entry point to evaluate graph-level echo chamber metrics."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Iterable

from src.evaluation.graph_base import calculate_modularity, compute_basic_metrics
from src.evaluation.graph_data import load_like_graph_from_connection


LATEST_RUN_FILE = Path("results/latest_run.txt")


def compute_metrics(sqlite_path: Path | str) -> None:
    conn = sqlite3.connect(sqlite_path)
    try:
        nodes, edges = load_like_graph_from_connection(conn)
    finally:
        conn.close()

    metrics = compute_basic_metrics(nodes, edges)
    modularity_score = calculate_modularity(nodes, edges)

    print("Graph metrics (like-based user network):")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    print(f"  modularity: {modularity_score:.4f}")


def resolve_sqlite_path(sqlite_path: str | None, run_dir: str | None) -> Path:
    if sqlite_path:
        return Path(sqlite_path)
    if run_dir:
        return Path(run_dir) / "simulation.db"
    if LATEST_RUN_FILE.exists():
        latest_dir = Path(LATEST_RUN_FILE.read_text().strip())
        candidate = latest_dir / "simulation.db"
        if candidate.exists():
            return candidate
    return Path("data/twitter_simulation.db")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute graph metrics from the simulation database.")
    parser.add_argument("--sqlite-path", help="Path to the simulation SQLite database.")
    parser.add_argument("--run-dir", help="Experiment run directory holding simulation.db.")

    args = parser.parse_args(list(argv) if argv is not None else None)

    path = resolve_sqlite_path(args.sqlite_path, args.run_dir)
    compute_metrics(path)


if __name__ == "__main__":
	main()