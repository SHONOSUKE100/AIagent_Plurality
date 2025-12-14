"""Utilities for embedding simulation content and storing it in SQLite."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Iterable, Sequence

from openai import AsyncOpenAI


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def _ensure_embedding_tables(db_path: Path) -> None:
    """Create embedding tables if they do not already exist."""

    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS post_embedding (
                post_id INTEGER PRIMARY KEY,
                model TEXT NOT NULL,
                embedding TEXT NOT NULL,
                embedded_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS comment_embedding (
                comment_id INTEGER PRIMARY KEY,
                model TEXT NOT NULL,
                embedding TEXT NOT NULL,
                embedded_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS user_embedding (
                user_id INTEGER PRIMARY KEY,
                model TEXT NOT NULL,
                embedding TEXT NOT NULL,
                embedded_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """
        )


def _load_text_items(
    db_path: Path,
    *,
    source_table: str,
    id_column: str,
    text_column: str,
    embedding_table: str,
    model: str,
    force: bool,
) -> list[tuple[int, str]]:
    """Return (id, text) pairs that still need embeddings."""

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        existing: set[int] = set()
        if not force:
            rows = conn.execute(
                f"SELECT {id_column} FROM {embedding_table} WHERE model = ?", (model,)
            ).fetchall()
            existing = {int(row[id_column]) for row in rows}

        rows = conn.execute(
            f"""
            SELECT {id_column}, {text_column}
            FROM {source_table}
            WHERE {text_column} IS NOT NULL AND TRIM({text_column}) != ''
            """
        ).fetchall()

    items = []
    for row in rows:
        item_id = int(row[id_column])
        if not force and item_id in existing:
            continue
        text = str(row[text_column]).strip()
        if text:
            items.append((item_id, text))
    return items


async def _embed_batch(
    client: AsyncOpenAI,
    texts: Sequence[str],
    *,
    model: str,
    max_retries: int = 3,
    retry_base_seconds: float = 2.0,
) -> list[list[float]]:
    """Embed a batch with simple exponential backoff on failure."""

    attempt = 0
    while True:
        try:
            response = await client.embeddings.create(model=model, input=list(texts))
            # Preserve caller order using the index field.
            data = sorted(response.data, key=lambda d: d.index)
            return [item.embedding for item in data]
        except Exception:
            attempt += 1
            if attempt > max_retries:
                raise
            delay = retry_base_seconds * (2 ** (attempt - 1))
            await asyncio.sleep(delay)


async def _embed_and_store(
    db_path: Path,
    *,
    items: Iterable[tuple[int, str]],
    model: str,
    batch_size: int,
    table: str,
    id_column: str,
    label: str,
) -> int:
    """Embed ``items`` and persist results. Returns number of embeddings written."""

    items_list = list(items)
    if not items_list:
        return 0

    client = AsyncOpenAI()
    written = 0
    for start in range(0, len(items_list), batch_size):
        batch = items_list[start : start + batch_size]
        texts = [text for _, text in batch]
        embeddings = await _embed_batch(client, texts, model=model)

        payload = [
            (item_id, model, json.dumps(embedding))
            for (item_id, _), embedding in zip(batch, embeddings, strict=True)
        ]

        with sqlite3.connect(db_path) as conn:
            conn.executemany(
                f"""
                INSERT OR REPLACE INTO {table} ({id_column}, model, embedding, embedded_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """,
                payload,
            )
            conn.commit()
        written += len(payload)
        print(f"[embedding] Stored {len(payload)} {label} embeddings (total {written})")

    return written


async def generate_text_embeddings(
    database_path: Path | str,
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
    force: bool = False,
) -> None:
    """Embed posts, comments, and bios into SQLite for downstream use."""

    db_path = Path(database_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    _ensure_embedding_tables(db_path)

    post_items = _load_text_items(
        db_path,
        source_table="post",
        id_column="post_id",
        text_column="content",
        embedding_table="post_embedding",
        model=model,
        force=force,
    )
    comment_items = _load_text_items(
        db_path,
        source_table="comment",
        id_column="comment_id",
        text_column="content",
        embedding_table="comment_embedding",
        model=model,
        force=force,
    )
    bio_items = _load_text_items(
        db_path,
        source_table="user",
        id_column="user_id",
        text_column="bio",
        embedding_table="user_embedding",
        model=model,
        force=force,
    )

    print(
        f"[embedding] Queueing embeddings with model='{model}': "
        f"{len(post_items)} posts, {len(comment_items)} comments, {len(bio_items)} bios"
    )

    await _embed_and_store(
        db_path,
        items=post_items,
        model=model,
        batch_size=batch_size,
        table="post_embedding",
        id_column="post_id",
        label="post",
    )
    await _embed_and_store(
        db_path,
        items=comment_items,
        model=model,
        batch_size=batch_size,
        table="comment_embedding",
        id_column="comment_id",
        label="comment",
    )
    await _embed_and_store(
        db_path,
        items=bio_items,
        model=model,
        batch_size=batch_size,
        table="user_embedding",
        id_column="user_id",
        label="bio",
    )
