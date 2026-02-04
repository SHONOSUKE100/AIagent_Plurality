"""Simulation entry points for running the OASIS environment."""

from __future__ import annotations

import asyncio
import csv
import json
import os
import random
import re
import sqlite3
import shutil
from pathlib import Path
from typing import Sequence

import oasis
from oasis import ActionType, DefaultPlatformType, LLMAction, ManualAction

from ..agents.agent_builder import (
    DEFAULT_AVAILABLE_ACTIONS,
    build_agent_graph,
    create_default_model,
    load_personas,
)
from oasis.social_agent.agent_environment import SocialEnvironment
from ..evaluation.graph_base import calculate_modularity, compute_basic_metrics
from ..evaluation.graph_data import load_like_graph_from_connection
from ..algorithms import RecommendationType, create_recommender
from camel.types import ModelType
from oasis.social_agent.agent import SocialAgent
from .embedding_store import (
    DEFAULT_EMBEDDING_MODEL,
    embed_selected_items,
    generate_text_embeddings,
)
from ..algorithms.contents_moderation import (
    Interaction as RecInteraction,
    Post as RecPost,
    User as RecUser,
)


_FAILFAST_INSTALLED = False
_SAFE_POSTS_PATCHED = False
_REBUILD_REC_EVERY = 2
_REBUILD_MAX_CANDIDATE_POSTS = 10000
_REBUILD_MAX_INTERACTIONS = 200000


def _is_rate_limit_error(exc: Exception) -> bool:
    """Best-effort detection for wrapped rate limit exceptions."""

    cur: Exception | None = exc
    for _ in range(5):
        if cur is None:
            break
        msg = str(cur).lower()
        if "rate limit" in msg or "rate_limit_exceeded" in msg or "error code: 429" in msg:
            return True
        cur = cur.__cause__ if isinstance(cur.__cause__, Exception) else None
    return False


async def _step_with_rate_limit_retry(
    env,
    actions: dict,
    *,
    context: str,
    initial_delay_sec: int,
    retry_delay_sec: int,
    max_retries: int,
) -> None:
    """Run env.step with long backoff to survive API rate limits."""

    def _clear_agent_memories_from_actions() -> None:
        # When a rate-limit (or any wrapped exception) interrupts a step,
        # some agents may already have recorded tool_calls in memory without
        # the corresponding tool responses. Clear before retrying.
        for agent in actions.keys():
            try:
                clear_memory = getattr(agent, "clear_memory", None)
                if callable(clear_memory):
                    clear_memory()
                    continue
                mem = getattr(agent, "memory", None)
                clear_fn = getattr(mem, "clear", None) if mem is not None else None
                if callable(clear_fn):
                    clear_fn()
            except Exception:
                continue

    attempt = 0
    delay = max(1, int(initial_delay_sec))
    while True:
        try:
            await env.step(actions)
            return
        except Exception as exc:  # noqa: BLE001 - rate limits may be wrapped
            if not _is_rate_limit_error(exc) or attempt >= max_retries:
                raise
            attempt += 1
            _clear_agent_memories_from_actions()
            mins = max(1, int(round(delay / 60)))
            print(
                f"[rate-limit] {context}: hit rate limit, sleeping {mins} min "
                f"(attempt {attempt}/{max_retries})"
            )
            await asyncio.sleep(delay)
            delay = max(1, int(retry_delay_sec))


def _enable_fail_fast_actions() -> None:
    """Patch SocialAgent to raise when LLM action returns an exception.

    OASIS currently logs and returns exceptions from perform_action_by_llm,
    which makes the simulation continue silently. We prefer fail-fast: if any
    agent action errors, bubble up to stop the run.
    """
    global _FAILFAST_INSTALLED
    if _FAILFAST_INSTALLED:
        return

    original = SocialAgent.perform_action_by_llm

    async def wrapper(self, *args, **kwargs):
        result = await original(self, *args, **kwargs)
        if isinstance(result, Exception):
            raise result
        return result

    SocialAgent.perform_action_by_llm = wrapper  # type: ignore[assignment]
    _FAILFAST_INSTALLED = True


def _patch_social_environment_posts(
    db_path: Path, *, max_posts: int = 200, max_content_chars: int = 280
) -> None:
    """Monkey-patch SocialEnvironment.get_posts_env to avoid calling action.refresh().

    refresh() triggers a heavy platform-side cache rebuild. Instead, read recent
    posts directly from SQLite. Prefer per-user recommendations from the ``rec``
    table so agents do not all see the same global timeline.
    """
    global _SAFE_POSTS_PATCHED
    if _SAFE_POSTS_PATCHED:
        return

    original_get_posts_env = SocialEnvironment.get_posts_env

    def _max_post_id() -> int:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT MAX(post_id) FROM post").fetchone()
            return int(row[0]) if row and row[0] is not None else 0

    def _rec_cache_key(user_id: int) -> int:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT MAX(rowid) FROM rec WHERE user_id = ?",
                (int(user_id),),
            ).fetchone()
            return int(row[0]) if row and row[0] is not None else 0

    def _format_rows(rows: list[sqlite3.Row]) -> list[dict]:
        posts: list[dict] = []
        for row in rows:
            content = row["content"] or ""
            if max_content_chars and len(content) > max_content_chars:
                content = content[:max_content_chars] + "..."
            posts.append(
                {
                    "post_id": str(row["post_id"]),
                    "user_id": int(row["user_id"]),
                    "content": content,
                    "created_at": str(row["created_at"]),
                    "num_likes": int(row["num_likes"] or 0),
                    "num_dislikes": int(row["num_dislikes"] or 0),
                }
            )
        posts.reverse()
        return posts

    def _fetch_recommended_posts(user_id: int, limit: int) -> list[dict]:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT p.post_id, p.user_id, p.content, p.created_at, p.num_likes, p.num_dislikes
                FROM rec r
                JOIN post p ON p.post_id = r.post_id
                WHERE r.user_id = ?
                ORDER BY p.post_id DESC
                LIMIT ?
                """,
                (int(user_id), int(limit)),
            ).fetchall()
        return _format_rows(rows)

    def _fetch_recent_posts() -> list[dict]:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT post_id, user_id, content, created_at, num_likes, num_dislikes
                FROM post
                ORDER BY post_id DESC
                LIMIT ?
                """,
                (int(max_posts),),
            ).fetchall()

        return _format_rows(rows)

    async def safe_get_posts_env(self) -> str:
        try:
            latest = _max_post_id()
            agent_id = int(getattr(self.action, "agent_id", -1))
            rec_key = _rec_cache_key(agent_id) if agent_id >= 0 else 0
            cache_key = getattr(self, "_posts_env_cache_key", None)
            full_key = (agent_id, latest, rec_key)
            if cache_key == full_key and hasattr(self, "_posts_env_cache_value"):
                return getattr(self, "_posts_env_cache_value")

            posts = (
                _fetch_recommended_posts(agent_id, max_posts)
                if agent_id >= 0
                else []
            )
            if not posts:
                posts = _fetch_recent_posts()
            if not posts:
                env_text = "There are no existing posts."
            else:
                posts_env = json.dumps(posts, ensure_ascii=False, indent=2)
                env_text = self.posts_env_template.substitute(posts=posts_env)

            setattr(self, "_posts_env_cache_key", full_key)
            setattr(self, "_posts_env_cache_value", env_text)
            return env_text
        except Exception:
            try:
                return await original_get_posts_env(self)
            except Exception:
                return "There are no existing posts."

    SocialEnvironment.get_posts_env = safe_get_posts_env  # type: ignore[assignment]
    _SAFE_POSTS_PATCHED = True


def _enable_sqlite_wal(db_path: Path) -> None:
    """Best-effort WAL + busy timeout to reduce SQLite lock errors."""

    try:
        with sqlite3.connect(db_path, timeout=30) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=30000;")
    except Exception:
        pass


def _install_recommender(platform, recommender) -> bool:
    """Try to attach our recommender to the platform; return True if successful."""
    attr_names = ["recommender", "rec_sys", "rec_system"]
    for name in attr_names:
        if hasattr(platform, name):
            try:
                setattr(platform, name, recommender)
                print(f"[recommender] attached to platform.{name}")
                return True
            except Exception:
                continue
    print(f"[recommender] could not attach; platform attrs: {dir(platform)}")
    return False


def _disable_platform_cache_refresh(platform) -> None:
    """Monkey patch cache refresh hooks on the platform/recommender to no-op."""

    import inspect

    async def _noop_async(*args, **kwargs):
        return None

    def _noop(*args, **kwargs):
        return None

    target_names = {
        "refresh_recommendation_system_cache",
        "refresh_recommendation_cache",
        "refresh_cache",
        "refresh",
        "update_rec_table",
    }

    def patch_obj(obj, prefix: str):
        for name in target_names:
            if hasattr(obj, name):
                try:
                    fn = getattr(obj, name)
                    is_async = inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(
                        getattr(type(obj), name, None)
                    )
                    if is_async:
                        setattr(obj, name, _noop_async)
                    else:
                        setattr(obj, name, _noop)
                    print(f"[cache] disabled {prefix}.{name}")
                except Exception:
                    pass

    patch_obj(platform, "platform")

    for nested_name in ("social", "twitter", "client", "engine", "rec_sys", "rec_system", "recommender"):
        nested = getattr(platform, nested_name, None)
        if nested is not None:
            patch_obj(nested, f"platform.{nested_name}")

    for nested_name in ("social", "twitter"):
        nested = getattr(platform, nested_name, None)
        if nested is None:
            continue
        for nested2 in ("twitter", "platform", "client", "engine", "rec_sys", "rec_system"):
            obj2 = getattr(nested, nested2, None)
            if obj2 is not None:
                patch_obj(obj2, f"platform.{nested_name}.{nested2}")


def _rebuild_rec_table(
    db_path: Path,
    recommender,
    max_recommendations: int | None = None,
    *,
    max_users: int | None = None,
    max_candidate_posts: int = 10000,
    max_recent_interactions: int = 200000,
    exposure_cap_per_post: int | None = None,
    exposure_cap_ratio: float = 0.2,
) -> None:
    """Recompute and persist rec table using the injected recommender (capped)."""

    conn: sqlite3.Connection | None = None
    try:
        import numpy as np  # Local import to avoid hard dependency for skip_embeddings

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA cache_size=-200000;")

        users_query = "SELECT user_id, bio FROM user"
        params: list[int] = []
        if max_users is not None:
            users_query += " LIMIT ?"
            params.append(int(max_users))
        users_rows = conn.execute(users_query, params).fetchall()

        rec_name = recommender.__class__.__name__.lower()
        needs_embeddings = ("collaborative" in rec_name) or ("bridging" in rec_name)

        embed_rows: dict[int, str] = {}
        if needs_embeddings:
            embed_rows = {
                int(row["user_id"]): row["embedding"]
                for row in conn.execute("SELECT user_id, embedding FROM user_embedding")
            }
        users: list[RecUser] = []
        for row in users_rows:
            uid = int(row["user_id"])
            embedding = None
            if needs_embeddings and uid in embed_rows:
                try:
                    embedding = np.array(json.loads(embed_rows[uid]), dtype=np.float32)
                except Exception:
                    embedding = None
            users.append(RecUser(user_id=uid, bio=row["bio"] or "", embedding=embedding))

        post_rows = conn.execute(
            """
            SELECT post_id, user_id, created_at, num_likes, num_dislikes
            FROM post
            ORDER BY post_id DESC
            LIMIT ?
            """,
            (int(max_candidate_posts),),
        ).fetchall()
        post_embed_map: dict[int, str] = {}
        if needs_embeddings:
            post_embed_map = {
                int(row["post_id"]): row["embedding"]
                for row in conn.execute(
                    """
                    SELECT post_id, embedding
                    FROM post_embedding
                    WHERE post_id IN (
                        SELECT post_id FROM post ORDER BY post_id DESC LIMIT ?
                    )
                    """,
                    (int(max_candidate_posts),),
                )
            }
        posts: list[RecPost] = []
        for row in post_rows:
            pid = int(row["post_id"])
            emb = None
            if needs_embeddings and pid in post_embed_map:
                try:
                    emb = np.array(json.loads(post_embed_map[pid]), dtype=np.float32)
                except Exception:
                    emb = None
            posts.append(
                RecPost(
                    post_id=str(pid),
                    user_id=int(row["user_id"]),
                    content="",
                    created_at=str(row["created_at"]),
                    num_likes=int(row["num_likes"] or 0),
                    num_dislikes=int(row["num_dislikes"] or 0),
                    embedding=emb,
                )
            )

        interactions: list[RecInteraction] = []
        like_rows = conn.execute(
            """
            SELECT user_id, post_id, created_at
            FROM 'like'
            ORDER BY rowid DESC
            LIMIT ?
            """,
            (int(max_recent_interactions),),
        ).fetchall()
        for row in like_rows:
            interactions.append(
                RecInteraction(
                    user_id=int(row["user_id"]),
                    post_id=str(row["post_id"]),
                    action="like",
                    timestamp=str(row["created_at"]),
                )
            )
        comment_rows = conn.execute(
            """
            SELECT user_id, post_id, created_at
            FROM comment
            ORDER BY rowid DESC
            LIMIT ?
            """,
            (int(max_recent_interactions),),
        ).fetchall()
        for row in comment_rows:
            interactions.append(
                RecInteraction(
                    user_id=int(row["user_id"]),
                    post_id=str(row["post_id"]),
                    action="comment",
                    timestamp=str(row["created_at"]),
                )
            )

        if not users or not posts:
            return

        rows = []
        limit = max_recommendations or getattr(recommender, "max_recommendations", 10)
        num_users = len(users)
        cap = exposure_cap_per_post
        if cap is None:
            cap = max(limit, int(num_users * max(0.0, exposure_cap_ratio)))
        cap = max(limit, int(cap))
        exposure_counts: dict[int, int] = {}
        for user in users:
            try:
                recs = recommender.recommend(user, posts, interactions, users)
            except Exception:
                recs = []
            if not recs:
                continue

            picked: list[int] = []
            for pid_str in recs:
                if len(picked) >= limit:
                    break
                pid = int(pid_str)
                if exposure_counts.get(pid, 0) >= cap:
                    continue
                picked.append(pid)
                exposure_counts[pid] = exposure_counts.get(pid, 0) + 1

            # Backfill if cap filtered too aggressively.
            if len(picked) < limit:
                remaining = [
                    int(p.post_id)
                    for p in posts
                    if int(p.post_id) not in set(picked) and exposure_counts.get(int(p.post_id), 0) < cap
                ]
                random.shuffle(remaining)
                for pid in remaining:
                    if len(picked) >= limit:
                        break
                    picked.append(pid)
                    exposure_counts[pid] = exposure_counts.get(pid, 0) + 1

            for pid in picked[:limit]:
                rows.append((user.user_id, pid))

        with conn:
            conn.execute("DELETE FROM rec")
            if rows:
                conn.executemany("INSERT INTO rec (user_id, post_id) VALUES (?, ?)", rows)
    except Exception as exc:
        print(f"[recommender] failed to rebuild rec table: {exc}")
    finally:
        if conn:
            conn.close()


def _prune_agent_memory(agent, keep_last: int = 10) -> None:
    """Trim agent memory to the most recent items to keep prompts small."""

    try:
        mem = getattr(agent, "memory", None)
        if mem is None:
            return

        # Common case: memory.messages is a list-like container
        messages = getattr(mem, "messages", None)
        if isinstance(messages, list):
            mem.messages = messages[-keep_last:]
            return

        # Fallback: if there's a clear method and no known structure, just clear
        clear_fn = getattr(mem, "clear", None)
        if callable(clear_fn):
            clear_fn()
    except Exception:
        # Swallow any unexpected memory structure issues; better to proceed than fail the run
        pass


def _capture_step_metrics(
    db_path: Path,
    metrics_path: Path | str | None,
    label: str,
    round_num: int | None = None,
    *,
    recommendation_type: RecommendationType | str | None = None,
    agent_action_ratio: float | None = None,
    warmup_random_rounds: int | None = None,
    active_recommender: str | None = None,
    warmup_phase: bool | None = None,
) -> None:
    """Compute graph metrics and append to CSV for the current step."""

    if metrics_path is None:
        return

    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        # Basic counts
        def _scalar(query: str) -> int:
            row = conn.execute(query).fetchone()
            return int(row[0]) if row and row[0] is not None else 0

        user_count = _scalar("SELECT COUNT(*) FROM user")
        post_count = _scalar("SELECT COUNT(*) FROM post")
        comment_count = _scalar("SELECT COUNT(*) FROM comment")
        like_count = _scalar("SELECT COUNT(*) FROM like")
        follow_count = _scalar("SELECT COUNT(*) FROM follow")

        # Graph metrics based on like network
        nodes, edges = load_like_graph_from_connection(conn)
        graph_metrics = compute_basic_metrics(nodes, edges)
        modularity_score = calculate_modularity(nodes, edges)

    fieldnames = [
        "label",
        "round",
        "recommendation_type",
        "active_recommender",
        "warmup_random_rounds",
        "warmup_phase",
        "agent_action_ratio",
        "users",
        "posts",
        "comments",
        "likes",
        "follows",
        "density",
        "average_clustering",
        "transitivity",
        "connected_components",
        "largest_component_ratio",
        "avg_shortest_path_length",
        "modularity",
    ]

    row = {
        "label": label,
        "round": round_num if round_num is not None else "",
        "recommendation_type": str(recommendation_type) if recommendation_type is not None else "",
        "active_recommender": active_recommender or "",
        "warmup_random_rounds": warmup_random_rounds if warmup_random_rounds is not None else "",
        "warmup_phase": int(bool(warmup_phase)) if warmup_phase is not None else "",
        "agent_action_ratio": agent_action_ratio if agent_action_ratio is not None else "",
        "users": user_count,
        "posts": post_count,
        "comments": comment_count,
        "likes": like_count,
        "follows": follow_count,
        "density": f"{graph_metrics['density']:.6f}",
        "average_clustering": f"{graph_metrics['average_clustering']:.6f}",
        "transitivity": f"{graph_metrics['transitivity']:.6f}",
        "connected_components": f"{graph_metrics['connected_components']:.6f}",
        "largest_component_ratio": f"{graph_metrics['largest_component_ratio']:.6f}",
        "avg_shortest_path_length": f"{graph_metrics['avg_shortest_path_length']:.6f}",
        "modularity": f"{modularity_score:.6f}",
    }

    write_header = not metrics_path.exists()
    with metrics_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _snapshot_database(
    db_path: Path,
    snapshot_dir: Path | str | None,
    label: str,
) -> None:
    """Copy the SQLite DB to a snapshot directory with the given label."""

    if snapshot_dir is None:
        return

    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    destination = snapshot_dir / f"{label}.db"
    shutil.copy2(db_path, destination)


def _infer_round_from_snapshot_path(snapshot_path: Path) -> int | None:
    """Infer the completed round number from a snapshot filename."""

    name = snapshot_path.name
    if name == "after_seeding.db":
        return 0
    match = re.fullmatch(r"round_(\d+)\.db", name)
    if match:
        return int(match.group(1))
    return None


def _truncate_step_metrics(metrics_path: Path | str | None, max_round: int) -> None:
    """Remove metrics rows beyond ``max_round`` to avoid duplicates on resume."""

    if metrics_path is None:
        return

    path = Path(metrics_path)
    if not path.exists():
        return

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not fieldnames:
        return

    def _round_ok(value: str | None) -> bool:
        if value is None or value == "":
            # Keep non-round markers like after_seeding.
            return True
        try:
            return int(float(value)) <= max_round
        except ValueError:
            return True

    kept_rows = [row for row in rows if _round_ok(row.get("round"))]

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)


def _get_max_time_step(db_path: Path) -> int:
    """Best-effort recovery of the platform time_step from existing tables."""

    queries = [
        "SELECT MAX(created_at) FROM user",
        "SELECT MAX(created_at) FROM post",
        "SELECT MAX(created_at) FROM comment",
        "SELECT MAX(created_at) FROM 'like'",
        "SELECT MAX(created_at) FROM follow",
    ]

    max_value = 0
    with sqlite3.connect(db_path) as conn:
        for query in queries:
            try:
                row = conn.execute(query).fetchone()
                if row and row[0] is not None:
                    max_value = max(max_value, int(float(row[0])))
            except sqlite3.Error:
                continue
            except (TypeError, ValueError):
                continue
    return max_value


def _bind_agents_to_channel(agent_graph, channel) -> None:
    """Ensure all agents use the environment's shared channel."""

    try:
        agent_items = list(agent_graph.get_agents())
    except Exception:
        return

    for _, agent in agent_items:
        try:
            agent.channel = channel
            env = getattr(agent, "env", None)
            action = getattr(env, "action", None) if env is not None else None
            if action is not None:
                action.channel = channel
        except Exception:
            continue


def load_seeding_data(seeding_path: Path | str) -> list[dict]:
    """Load seeding posts from a JSON file.

    Raises an explicit error when the path is missing or contains no items so
    the simulation does not silently run with an empty timeline.
    """
    path = Path(seeding_path)
    if not path.exists():
        raise FileNotFoundError(f"Seeding file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list) or not data:
        raise ValueError(f"Seeding file must be a non-empty list: {path}")

    return data


async def run_simulation(
    *,
    persona_path: Path | str,
    database_path: Path | str,
    seeding_path: Path | str = Path("data/seeding.json"),
    seed_post_count: int = 20,
    llm_rounds: int = 1,
    agent_action_ratio: float = 0.3,
    recommendation_type: RecommendationType | str = RecommendationType.RANDOM,
    platform: DefaultPlatformType = DefaultPlatformType.TWITTER,
    available_actions: Sequence[ActionType] | None = None,
    model_type: str | ModelType = "gpt-4o",
    model_temperature: float | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_batch_size: int = 32,
    skip_embeddings: bool = False,
    step_metrics_path: Path | str | None = None,
    step_snapshot_dir: Path | str | None = None,
    max_memory_messages: int = 5,
    warmup_random_rounds: int = 10,
    resume_from_snapshot: Path | str | None = None,
    resume_last_completed_round: int | None = None,
    rate_limit_initial_delay_sec: int = 600,
    rate_limit_retry_delay_sec: int = 300,
    max_rate_limit_retries: int = 50,
) -> None:
    """Run the social simulation once using personas from ``persona_path``.
    
    Args:
        persona_path: Path to the persona JSON file.
        database_path: Path to the SQLite database file.
        seeding_path: Path to the seeding JSON file for initial posts.
        seed_post_count: Number of seeding posts to create initially.
        llm_rounds: Number of LLM-driven action rounds.
        agent_action_ratio: Ratio of agents (0.0-1.0) to randomly select for each LLM round.
        recommendation_type: Type of recommendation algorithm to use.
            Options: "random", "collaborative", "bridging"
        platform: The platform type for the simulation.
        available_actions: Available action types for agents.
        embedding_model: Embedding model name to store text vectors.
        embedding_batch_size: Batch size for embedding API calls.
        skip_embeddings: Skip embedding step (useful to avoid API calls).
        step_metrics_path: Optional CSV path to record per-step graph metrics.
        step_snapshot_dir: Optional directory to save SQLite snapshots after each step.
        max_memory_messages: Number of past messages to retain in agent memory (smaller saves RAM).
        warmup_random_rounds: Use random recommendations for the first N rounds
            before switching to the requested algorithm to inject exploration
            and reduce early collapse.
        resume_from_snapshot: Optional snapshot DB path to resume from.
        resume_last_completed_round: The last fully completed round in the snapshot.
        rate_limit_initial_delay_sec: Initial sleep when rate limited (default 10 minutes).
        rate_limit_retry_delay_sec: Subsequent sleep between retries (default 5 minutes).
        max_rate_limit_retries: Maximum retries before failing the run.

    Notes:
        - Embeddings are generated incrementally (bios after reset, seeded posts
          after seeding, and new posts/comments after each round) so downstream
          evaluations like ECS can run per step, even for random recommendations.
    """

    if embedding_batch_size <= 0:
        raise ValueError("embedding_batch_size must be positive")

    incremental_embeddings = not skip_embeddings

    _enable_fail_fast_actions()
    personas = load_personas(persona_path)
    model = create_default_model(model_type=model_type, temperature=model_temperature)
    agent_graph = build_agent_graph(personas, model, available_actions=available_actions)

    # Load seeding data
    seeding_data = load_seeding_data(seeding_path)

    # Create the recommender for custom content moderation
    recommender_main = create_recommender(recommendation_type)
    warmup_recommender = None
    if warmup_random_rounds > 0 and recommendation_type != RecommendationType.RANDOM:
        warmup_recommender = create_recommender(RecommendationType.RANDOM)
    resume_mode = resume_from_snapshot is not None
    resume_snapshot_path = Path(resume_from_snapshot) if resume_from_snapshot else None

    inferred_resume_round = (
        _infer_round_from_snapshot_path(resume_snapshot_path)
        if resume_snapshot_path is not None
        else None
    )
    resume_round = (
        int(resume_last_completed_round)
        if resume_last_completed_round is not None
        else inferred_resume_round
    )
    if resume_mode:
        if resume_snapshot_path is None or not resume_snapshot_path.exists():
            raise FileNotFoundError(f"Resume snapshot not found: {resume_snapshot_path}")
        if resume_round is None:
            raise ValueError(
                "Could not infer resume round; please provide resume_last_completed_round."
            )
        if resume_round < 0:
            raise ValueError("resume_last_completed_round must be non-negative")
        if resume_round > llm_rounds:
            raise ValueError(
                f"resume_last_completed_round ({resume_round}) exceeds llm_rounds ({llm_rounds})"
            )

    if warmup_recommender and resume_round is not None and resume_round >= warmup_random_rounds:
        active_recommender = recommender_main
    else:
        active_recommender = warmup_recommender or recommender_main

    db_path = Path(database_path)
    if resume_mode:
        if db_path.exists():
            db_path.unlink()
        shutil.copy2(resume_snapshot_path, db_path)
        _truncate_step_metrics(step_metrics_path, max_round=resume_round)
    else:
        if db_path.exists():
            db_path.unlink()

    env = oasis.make(
        agent_graph=agent_graph,
        platform=platform,
        database_path=str(db_path),
    )
    _enable_sqlite_wal(db_path)

    # Patch OASIS SocialEnvironment to avoid heavy refreshes for posts_env.
    _patch_social_environment_posts(db_path, max_posts=200, max_content_chars=280)
    # Inject custom recommender into the platform to avoid the heavy default cache refresh.
    _install_recommender(env.platform, active_recommender)
    _disable_platform_cache_refresh(env.platform)
    _rebuild_rec_table(
        db_path,
        active_recommender,
        max_candidate_posts=_REBUILD_MAX_CANDIDATE_POSTS,
        max_recent_interactions=_REBUILD_MAX_INTERACTIONS,
    )

    try:
        if resume_mode:
            env.platform_task = asyncio.create_task(env.platform.running())
            _bind_agents_to_channel(env.agent_graph, env.channel)
            # Ensure no dangling tool_calls remain in memory from a prior
            # interrupted step before we resume.
            try:
                for _, agent in env.agent_graph.get_agents():
                    clear_memory = getattr(agent, "clear_memory", None)
                    if callable(clear_memory):
                        clear_memory()
                        continue
                    mem = getattr(agent, "memory", None)
                    clear_fn = getattr(mem, "clear", None) if mem is not None else None
                    if callable(clear_fn):
                        clear_fn()
            except Exception:
                pass
            try:
                env.platform.sandbox_clock.time_step = _get_max_time_step(db_path)
            except Exception:
                pass
            _rebuild_rec_table(
                db_path,
                active_recommender,
                max_candidate_posts=_REBUILD_MAX_CANDIDATE_POSTS,
                max_recent_interactions=_REBUILD_MAX_INTERACTIONS,
            )
        else:
            await env.reset()
            if incremental_embeddings:
                await _embed_all_bios(
                    db_path=db_path,
                    model=embedding_model,
                    batch_size=embedding_batch_size,
                )

            last_post_id = _get_max_id(db_path, "post", "post_id")
            await _seed_posts(
                env,
                seeding_data,
                seed_post_count,
                rate_limit_initial_delay_sec=rate_limit_initial_delay_sec,
                rate_limit_retry_delay_sec=rate_limit_retry_delay_sec,
                max_rate_limit_retries=max_rate_limit_retries,
            )
            if incremental_embeddings:
                new_posts = _get_new_ids_since(db_path, "post", "post_id", last_post_id)
                await embed_selected_items(
                    database_path=db_path,
                    post_ids=new_posts,
                    model=embedding_model,
                    batch_size=embedding_batch_size,
                )

            _capture_step_metrics(
                db_path=db_path,
                metrics_path=step_metrics_path,
                label="after_seeding",
                round_num=0,
                recommendation_type=recommendation_type,
                agent_action_ratio=agent_action_ratio,
                warmup_random_rounds=warmup_random_rounds,
                active_recommender=active_recommender.__class__.__name__,
                warmup_phase=bool(warmup_recommender and warmup_random_rounds > 0),
            )
            _rebuild_rec_table(
                db_path,
                active_recommender,
                max_candidate_posts=_REBUILD_MAX_CANDIDATE_POSTS,
                max_recent_interactions=_REBUILD_MAX_INTERACTIONS,
            )
            _snapshot_database(
                db_path=db_path,
                snapshot_dir=step_snapshot_dir,
                label="after_seeding",
            )

        start_round = resume_round if resume_round is not None else 0
        for round_num in range(start_round, llm_rounds):
            # Switch from warmup random to target recommender after warmup_random_rounds
            if warmup_recommender and round_num >= warmup_random_rounds:
                if active_recommender is not recommender_main:
                    active_recommender = recommender_main
                    _install_recommender(env.platform, active_recommender)
                    _rebuild_rec_table(
                        db_path,
                        active_recommender,
                        max_candidate_posts=_REBUILD_MAX_CANDIDATE_POSTS,
                        max_recent_interactions=_REBUILD_MAX_INTERACTIONS,
                    )
            await _llm_round(
                env,
                agent_action_ratio,
                active_recommender,
                round_num,
                incremental_embeddings=incremental_embeddings,
                db_path=db_path,
                embedding_model=embedding_model,
                embedding_batch_size=embedding_batch_size,
                max_memory_messages=max_memory_messages,
                rate_limit_initial_delay_sec=rate_limit_initial_delay_sec,
                rate_limit_retry_delay_sec=rate_limit_retry_delay_sec,
                max_rate_limit_retries=max_rate_limit_retries,
            )
            if (round_num + 1) % _REBUILD_REC_EVERY == 0 or (round_num + 1 == llm_rounds):
                _rebuild_rec_table(
                    db_path,
                    active_recommender,
                    max_candidate_posts=_REBUILD_MAX_CANDIDATE_POSTS,
                    max_recent_interactions=_REBUILD_MAX_INTERACTIONS,
                )
            _capture_step_metrics(
                db_path=db_path,
                metrics_path=step_metrics_path,
                label=f"round_{round_num + 1}",
                round_num=round_num + 1,
                recommendation_type=recommendation_type,
                agent_action_ratio=agent_action_ratio,
                warmup_random_rounds=warmup_random_rounds,
                active_recommender=active_recommender.__class__.__name__,
                warmup_phase=bool(warmup_recommender and round_num < warmup_random_rounds),
            )
            _snapshot_database(
                db_path=db_path,
                snapshot_dir=step_snapshot_dir,
                label=f"round_{round_num + 1}",
            )
    finally:
        await env.close()

    if not skip_embeddings:
        await generate_text_embeddings(
            database_path=db_path,
            model=embedding_model,
            batch_size=embedding_batch_size,
        )

    _capture_step_metrics(
        db_path=db_path,
        metrics_path=step_metrics_path,
        label="final",
        round_num=llm_rounds,
        recommendation_type=recommendation_type,
        agent_action_ratio=agent_action_ratio,
        warmup_random_rounds=warmup_random_rounds,
        active_recommender=active_recommender.__class__.__name__,
        warmup_phase=False,
    )
    _snapshot_database(
        db_path=db_path,
        snapshot_dir=step_snapshot_dir,
        label="final",
    )


async def _seed_posts(
    env,
    seeding_data: list[dict],
    seed_post_count: int,
    *,
    rate_limit_initial_delay_sec: int,
    rate_limit_retry_delay_sec: int,
    max_rate_limit_retries: int,
) -> None:
    """Create initial posts using seeding data.
    
    Each seeding post is assigned to a random agent who will create that post.
    After seeding, agent memories are cleared to prevent tool_calls state issues.
    """
    agent_items = list(env.agent_graph.get_agents())
    if not agent_items or not seeding_data:
        return

    # Use the minimum of seed_post_count, available agents, and seeding data
    actual_count = min(seed_post_count, len(agent_items), len(seeding_data))
    
    # Randomly select agents for seeding
    selected_agents = random.sample(agent_items, actual_count)
    # Randomly select seeding contents
    selected_seeds = random.sample(seeding_data, actual_count)

    actions = {}
    seeded_agents = []
    for (agent_id, agent), seed in zip(selected_agents, selected_seeds):
        content = seed.get("content", "")
        if content:
            actions[agent] = [
                ManualAction(
                    action_type=ActionType.CREATE_POST,
                    action_args={
                        "content": content,
                    },
                )
            ]
            seeded_agents.append(agent)

    if actions:
        await _step_with_rate_limit_retry(
            env,
            actions,
            context="seeding",
            initial_delay_sec=rate_limit_initial_delay_sec,
            retry_delay_sec=rate_limit_retry_delay_sec,
            max_retries=max_rate_limit_retries,
        )
        
        # Clear memory for seeded agents to prevent tool_calls state corruption
        # This fixes the OpenAI error: "An assistant message with 'tool_calls' 
        # must be followed by tool messages responding to each 'tool_call_id'"
        for agent in seeded_agents:
            try:
                agent.memory.clear()
            except Exception:
                pass  # Ignore if memory clearing fails


async def _llm_round(
    env,
    agent_action_ratio: float = 0.3,
    recommender=None,
    round_num: int = 0,
    clear_memory_before_action: bool = True,
    incremental_embeddings: bool = False,
    db_path: Path | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_batch_size: int = 32,
    max_memory_messages: int = 5,
    rate_limit_initial_delay_sec: int = 600,
    rate_limit_retry_delay_sec: int = 300,
    max_rate_limit_retries: int = 50,
) -> None:
    """Execute one LLM round with a random subset of agents.
    
    Args:
        env: The simulation environment.
        agent_action_ratio: Ratio of agents (0.0-1.0) to randomly select for this round.
        recommender: Optional custom recommender for content moderation.
        round_num: Current round number (for logging/debugging).
        clear_memory_before_action: Whether to clear agent memory before LLM action
            to prevent tool_calls state corruption.
        incremental_embeddings: Whether to embed new posts/comments immediately after creation.
        db_path: SQLite path for embedding writes (required if incremental_embeddings).
        embedding_model: Embedding model name.
        embedding_batch_size: Batch size for embedding API calls.
    """
    agent_items = list(env.agent_graph.get_agents())
    if not agent_items:
        return
    
    # Calculate how many agents to select based on the ratio
    num_to_select = max(1, int(len(agent_items) * agent_action_ratio))
    num_to_select = min(num_to_select, len(agent_items))
    
    # Randomly select agents for this round
    selected_agents = random.sample(agent_items, num_to_select)
    
    # Log the round info
    print(f"Round {round_num + 1}: {num_to_select}/{len(agent_items)} agents selected ({agent_action_ratio*100:.0f}%)")
    
    # Clear memory for selected agents to prevent tool_calls state issues
    if clear_memory_before_action:
        for _, agent in selected_agents:
            try:
                mem = getattr(agent, "memory", None)
                clear_fn = getattr(mem, "clear", None) if mem is not None else None
                if callable(clear_fn):
                    # Full clear is safer than pruning when tool_calls are present.
                    clear_fn()
                else:
                    _prune_agent_memory(agent, keep_last=max_memory_messages)
            except Exception:
                pass  # Ignore if memory clearing fails
    
    prev_post_id = _get_max_id(db_path, "post", "post_id") if incremental_embeddings and db_path else 0
    prev_comment_id = _get_max_id(db_path, "comment", "comment_id") if incremental_embeddings and db_path else 0

    actions = {
        agent: LLMAction()
        for _, agent in selected_agents
    }

    if actions:
        await _step_with_rate_limit_retry(
            env,
            actions,
            context=f"round {round_num + 1}",
            initial_delay_sec=rate_limit_initial_delay_sec,
            retry_delay_sec=rate_limit_retry_delay_sec,
            max_retries=max_rate_limit_retries,
        )
        if incremental_embeddings and db_path:
            new_post_ids = _get_new_ids_since(db_path, "post", "post_id", prev_post_id)
            new_comment_ids = _get_new_ids_since(
                db_path, "comment", "comment_id", prev_comment_id
            )
            await embed_selected_items(
                database_path=db_path,
                post_ids=new_post_ids,
                comment_ids=new_comment_ids,
                model=embedding_model,
                batch_size=embedding_batch_size,
            )


def _get_max_id(db_path: Path, table: str, id_column: str) -> int:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(f"SELECT MAX({id_column}) FROM {table}").fetchone()
        return int(row[0]) if row and row[0] is not None else 0


def _get_new_ids_since(
    db_path: Path, table: str, id_column: str, last_id: int
) -> list[int]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT {id_column} FROM {table} WHERE {id_column} > ? ORDER BY {id_column}",
            (last_id,),
        ).fetchall()
        return [int(row[0]) for row in rows]


async def _embed_all_bios(db_path: Path, model: str, batch_size: int) -> None:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT user_id FROM user").fetchall()
        ids = [int(row[0]) for row in rows]
    await embed_selected_items(
        database_path=db_path,
        user_ids=ids,
        model=model,
        batch_size=batch_size,
    )


def _is_random_recommendation(rec_type: RecommendationType | str) -> bool:
    if isinstance(rec_type, RecommendationType):
        return rec_type == RecommendationType.RANDOM
    return str(rec_type).lower() == RecommendationType.RANDOM.value


def run(
    persona_path: Path | str = Path("data/persona/persona.json"),
    database_path: Path | str = Path("data/twitter_simulation.db"),
    seeding_path: Path | str = Path("data/seeding.json"),
    seed_post_count: int = 10,
    llm_rounds: int = 1,
    agent_action_ratio: float = 0.2,
    recommendation_type: RecommendationType | str = RecommendationType.RANDOM,
    model_type: str | ModelType = "gpt-4o",
    model_temperature: float | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_batch_size: int = 16,
    skip_embeddings: bool = False,
    step_metrics_path: Path | str | None = None,
    step_snapshot_dir: Path | str | None = None,
    max_memory_messages: int = 5,
    warmup_random_rounds: int = 0,
    resume_from_snapshot: Path | str | None = None,
    resume_last_completed_round: int | None = None,
    rate_limit_initial_delay_sec: int = 600,
    rate_limit_retry_delay_sec: int = 300,
    max_rate_limit_retries: int = 50,
) -> None:
    """Convenience wrapper that mirrors the notebook execution.
    
    Args:
        persona_path: Path to the persona JSON file.
        database_path: Path to the SQLite database file.
        seeding_path: Path to the seeding JSON file for initial posts.
        seed_post_count: Number of seeding posts to create initially.
        llm_rounds: Number of LLM-driven action rounds.
        agent_action_ratio: Ratio of agents (0.0-1.0) to randomly select for each LLM round.
        recommendation_type: Type of recommendation algorithm to use.
            Options: "random", "collaborative", "bridging"
        embedding_model: Embedding model name to store text vectors.
        embedding_batch_size: Batch size for embedding API calls.
        skip_embeddings: Skip embedding step (useful to avoid API calls).
        step_metrics_path: Optional CSV path to record per-step graph metrics.
        step_snapshot_dir: Optional directory to save SQLite snapshots after each step.
        max_memory_messages: Number of past messages to retain in agent memory (smaller saves RAM).
        warmup_random_rounds: Number of initial rounds to use random recommendations before switching to the chosen recommender.
        rate_limit_initial_delay_sec: Initial sleep when rate limited (default 10 minutes).
        rate_limit_retry_delay_sec: Subsequent sleep between retries (default 5 minutes).
        max_rate_limit_retries: Maximum retries before failing the run.
    """

    asyncio.run(
        run_simulation(
            persona_path=persona_path,
            database_path=database_path,
            seeding_path=seeding_path,
            seed_post_count=seed_post_count,
            llm_rounds=llm_rounds,
            agent_action_ratio=agent_action_ratio,
            recommendation_type=recommendation_type,
            available_actions=DEFAULT_AVAILABLE_ACTIONS,
            model_type=model_type,
            model_temperature=model_temperature,
            embedding_model=embedding_model,
            embedding_batch_size=embedding_batch_size,
            skip_embeddings=skip_embeddings,
            step_metrics_path=step_metrics_path,
            step_snapshot_dir=step_snapshot_dir,
            max_memory_messages=max_memory_messages,
            warmup_random_rounds=warmup_random_rounds,
            resume_from_snapshot=resume_from_snapshot,
            resume_last_completed_round=resume_last_completed_round,
            rate_limit_initial_delay_sec=rate_limit_initial_delay_sec,
            rate_limit_retry_delay_sec=rate_limit_retry_delay_sec,
            max_rate_limit_retries=max_rate_limit_retries,
        )
    )
