"""Utilities for pushing the simulation SQLite data into Neo4j."""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping

from neo4j import GraphDatabase, basic_auth


def chunked(iterable: Iterable[Mapping], size: int) -> Iterator[List[Mapping]]:
    """Yield fixed-size batches from ``iterable``."""

    batch: List[Mapping] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _connect_sqlite(path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _connect_neo4j(uri: str, username: str, password: str):
    return GraphDatabase.driver(uri, auth=basic_auth(username, password))


def push_to_neo4j(
    *,
    sqlite_path: Path | str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    clear_database: bool = True,
    batch_size: int = 200,
) -> None:
    conn = _connect_sqlite(sqlite_path)
    driver = _connect_neo4j(neo4j_uri, neo4j_user, neo4j_password)

    try:
        with conn:
            with driver.session() as session:
                if clear_database:
                    session.run("MATCH (n) DETACH DELETE n")

                _import_users(conn, session, batch_size)
                _import_posts(conn, session, batch_size)
                _import_comments(conn, session, batch_size)
                _import_follow_edges(conn, session, batch_size)
                _import_mutes(conn, session, batch_size)
                _import_likes(conn, session, batch_size)
                _import_dislikes(conn, session, batch_size)
                _import_comment_likes(conn, session, batch_size)
                _import_comment_dislikes(conn, session, batch_size)
                _import_reports(conn, session, batch_size)
                _import_recommendations(conn, session, batch_size)
                _import_traces(conn, session, batch_size)
    finally:
        conn.close()
        driver.close()


def _rows(conn: sqlite3.Connection, query: str) -> List[sqlite3.Row]:
    return list(conn.execute(query))


def _import_users(conn: sqlite3.Connection, session, batch_size: int) -> None:
    rows = _rows(conn, "SELECT * FROM user")
    if not rows:
        return

    payload = [dict(row) for row in rows]
    query = (
        "UNWIND $rows AS row "
        "MERGE (u:User {user_id: row.user_id}) "
        "SET u.agent_id = row.agent_id, "
        "u.user_name = row.user_name, "
        "u.name = row.name, "
        "u.bio = row.bio, "
        "u.created_at = row.created_at, "
        "u.num_followings = row.num_followings, "
        "u.num_followers = row.num_followers"
    )

    for batch in chunked(payload, batch_size):
        session.run(query, rows=batch)


def _import_posts(conn: sqlite3.Connection, session, batch_size: int) -> None:
    rows = _rows(conn, "SELECT * FROM post")
    if not rows:
        return

    payload = [dict(row) for row in rows]
    query = (
        "UNWIND $rows AS row "
        "MERGE (p:Post {post_id: row.post_id}) "
        "SET p.content = row.content, "
        "p.quote_content = row.quote_content, "
        "p.created_at = row.created_at, "
        "p.num_likes = row.num_likes, "
        "p.num_dislikes = row.num_dislikes, "
        "p.num_shares = row.num_shares, "
        "p.num_reports = row.num_reports "
        "WITH row, p "
        "MATCH (author:User {user_id: row.user_id}) "
        "MERGE (author)-[:POSTED]->(p) "
        "WITH row, p "
        "WHERE row.original_post_id IS NOT NULL "
        "MATCH (original:Post {post_id: row.original_post_id}) "
        "MERGE (p)-[:QUOTED]->(original)"
    )

    for batch in chunked(payload, batch_size):
        session.run(query, rows=batch)


def _import_comments(conn: sqlite3.Connection, session, batch_size: int) -> None:
    rows = _rows(conn, "SELECT * FROM comment")
    if not rows:
        return

    payload = [dict(row) for row in rows]
    query = (
        "UNWIND $rows AS row "
        "MERGE (c:Comment {comment_id: row.comment_id}) "
        "SET c.content = row.content, "
        "c.created_at = row.created_at, "
        "c.num_likes = row.num_likes, "
        "c.num_dislikes = row.num_dislikes "
        "WITH row, c "
        "MATCH (author:User {user_id: row.user_id}) "
        "MERGE (author)-[:WROTE_COMMENT]->(c) "
        "WITH row, c "
        "MATCH (p:Post {post_id: row.post_id}) "
        "MERGE (c)-[:ON_POST]->(p)"
    )

    for batch in chunked(payload, batch_size):
        session.run(query, rows=batch)


def _import_follow_edges(conn: sqlite3.Connection, session, batch_size: int) -> None:
    rows = _rows(conn, "SELECT * FROM follow")
    if not rows:
        return

    payload = [dict(row) for row in rows]
    query = (
        "UNWIND $rows AS row "
        "MATCH (src:User {user_id: row.follower_id}) "
        "MATCH (dst:User {user_id: row.followee_id}) "
        "MERGE (src)-[r:FOLLOWED]->(dst) "
        "SET r.created_at = row.created_at"
    )

    for batch in chunked(payload, batch_size):
        session.run(query, rows=batch)


def _import_mutes(conn: sqlite3.Connection, session, batch_size: int) -> None:
    rows = _rows(conn, "SELECT * FROM mute")
    if not rows:
        return

    payload = [dict(row) for row in rows]
    query = (
        "UNWIND $rows AS row "
        "MATCH (src:User {user_id: row.muter_id}) "
        "MATCH (dst:User {user_id: row.mutee_id}) "
        "MERGE (src)-[r:MUTED]->(dst) "
        "SET r.created_at = row.created_at"
    )

    for batch in chunked(payload, batch_size):
        session.run(query, rows=batch)


def _import_likes(conn: sqlite3.Connection, session, batch_size: int) -> None:
    rows = _rows(conn, "SELECT * FROM like")
    if not rows:
        return

    payload = [dict(row) for row in rows]
    query = (
        "UNWIND $rows AS row "
        "MATCH (u:User {user_id: row.user_id}) "
        "MATCH (p:Post {post_id: row.post_id}) "
        "MERGE (u)-[r:LIKED]->(p) "
        "SET r.created_at = row.created_at"
    )

    for batch in chunked(payload, batch_size):
        session.run(query, rows=batch)


def _import_dislikes(conn: sqlite3.Connection, session, batch_size: int) -> None:
    rows = _rows(conn, "SELECT * FROM dislike")
    if not rows:
        return

    payload = [dict(row) for row in rows]
    query = (
        "UNWIND $rows AS row "
        "MATCH (u:User {user_id: row.user_id}) "
        "MATCH (p:Post {post_id: row.post_id}) "
        "MERGE (u)-[r:DISLIKED]->(p) "
        "SET r.created_at = row.created_at"
    )

    for batch in chunked(payload, batch_size):
        session.run(query, rows=batch)


def _import_comment_likes(conn: sqlite3.Connection, session, batch_size: int) -> None:
    rows = _rows(conn, "SELECT * FROM comment_like")
    if not rows:
        return

    payload = [dict(row) for row in rows]
    query = (
        "UNWIND $rows AS row "
        "MATCH (u:User {user_id: row.user_id}) "
        "MATCH (c:Comment {comment_id: row.comment_id}) "
        "MERGE (u)-[r:LIKED_COMMENT]->(c) "
        "SET r.created_at = row.created_at"
    )

    for batch in chunked(payload, batch_size):
        session.run(query, rows=batch)


def _import_comment_dislikes(conn: sqlite3.Connection, session, batch_size: int) -> None:
    rows = _rows(conn, "SELECT * FROM comment_dislike")
    if not rows:
        return

    payload = [dict(row) for row in rows]
    query = (
        "UNWIND $rows AS row "
        "MATCH (u:User {user_id: row.user_id}) "
        "MATCH (c:Comment {comment_id: row.comment_id}) "
        "MERGE (u)-[r:DISLIKED_COMMENT]->(c) "
        "SET r.created_at = row.created_at"
    )

    for batch in chunked(payload, batch_size):
        session.run(query, rows=batch)


def _import_reports(conn: sqlite3.Connection, session, batch_size: int) -> None:
    rows = _rows(conn, "SELECT * FROM report")
    if not rows:
        return

    payload = [dict(row) for row in rows]
    query = (
        "UNWIND $rows AS row "
        "MATCH (u:User {user_id: row.user_id}) "
        "MATCH (p:Post {post_id: row.post_id}) "
        "MERGE (u)-[r:REPORTED]->(p) "
        "SET r.reason = row.report_reason, "
        "r.created_at = row.created_at"
    )

    for batch in chunked(payload, batch_size):
        session.run(query, rows=batch)


def _import_recommendations(conn: sqlite3.Connection, session, batch_size: int) -> None:
    rows = _rows(conn, "SELECT * FROM rec")
    if not rows:
        return

    payload = [dict(row) for row in rows]
    query = (
        "UNWIND $rows AS row "
        "MATCH (u:User {user_id: row.user_id}) "
        "MATCH (p:Post {post_id: row.post_id}) "
        "MERGE (u)-[:RECOMMENDED]->(p)"
    )

    for batch in chunked(payload, batch_size):
        session.run(query, rows=batch)


def _import_traces(conn: sqlite3.Connection, session, batch_size: int) -> None:
    rows = _rows(conn, "SELECT * FROM trace")
    if not rows:
        return

    payload = [dict(row) for row in rows]
    query = (
        "UNWIND $rows AS row "
        "MERGE (t:Trace {user_id: row.user_id, created_at: row.created_at, action: row.action}) "
        "SET t.info = row.info "
        "WITH row, t "
        "MATCH (u:User {user_id: row.user_id}) "
        "MERGE (u)-[:HAS_TRACE]->(t)"
    )

    for batch in chunked(payload, batch_size):
        session.run(query, rows=batch)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the SQLite simulation data to Neo4j.")
    parser.add_argument("--sqlite-path", default="data/twitter_simulation.db", help="Path to the SQLite database generated by the simulation.")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687", help="Bolt URI of the Neo4j instance.")
    parser.add_argument("--neo4j-user", default="neo4j", help="Neo4j username.")
    parser.add_argument(
        "--neo4j-password",
        default=None,
        help="Neo4j password (defaults to $NEO4J_PASSWORD or 'test').",
    )
    parser.add_argument("--no-clear", action="store_true", help="Keep existing nodes and relationships instead of clearing the database first.")
    parser.add_argument("--batch-size", type=int, default=200, help="Number of rows pushed per Cypher query batch.")

    args = parser.parse_args()

    clear_database = not args.no_clear
    password = args.neo4j_password or os.getenv("NEO4J_PASSWORD") or "neo4j1234"

    push_to_neo4j(
        sqlite_path=Path(args.sqlite_path),
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=password,
        clear_database=clear_database,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
     main()
