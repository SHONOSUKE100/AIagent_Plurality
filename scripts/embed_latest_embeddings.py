"""Compute and store embeddings for the latest simulation database.

Usage:
    uv run scripts/embed_latest_embeddings.py
    uv run scripts/embed_latest_embeddings.py --db-path path/to/simulation.db --model text-embedding-3-small --batch-size 32
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.simulation.embedding_store import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    generate_text_embeddings,
)

LATEST_RUN_FILE = Path("results/latest_run.txt")
DEFAULT_DB = Path("data/twitter_simulation.db")


def resolve_db_path(db_path: str | None) -> Path:
    if db_path:
        return Path(db_path)
    if LATEST_RUN_FILE.exists():
        try:
            run_dir = Path(LATEST_RUN_FILE.read_text().strip())
            candidate = run_dir / "simulation.db"
            if candidate.exists():
                return candidate
        except OSError:
            pass
    return DEFAULT_DB


async def main_async(args: argparse.Namespace) -> None:
    target = resolve_db_path(args.db_path)
    if not target.exists():
        raise FileNotFoundError(f"Database not found: {target}")

    print(f"[embed] target DB: {target}")
    print(f"[embed] model={args.model} batch_size={args.batch_size} force={args.force}")
    await generate_text_embeddings(
        database_path=target,
        model=args.model,
        batch_size=args.batch_size,
        force=args.force,
    )
    print("[embed] done")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute embeddings for the latest simulation DB.")
    parser.add_argument("--db-path", help="Explicit path to simulation.db. Defaults to results/latest_run.txt -> simulation.db")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL, help="Embedding model name")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding API calls")
    parser.add_argument("--force", action="store_true", help="Re-embed even if embeddings already exist for the model")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
