"""Simulation entry points for running the OASIS environment."""

from __future__ import annotations

import asyncio
import csv
import json
import os
import random
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


_FAILFAST_INSTALLED = False


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
    recommender = create_recommender(recommendation_type)

    db_path = Path(database_path)
    if db_path.exists():
        db_path.unlink()

    env = oasis.make(
        agent_graph=agent_graph,
        platform=platform,
        database_path=str(db_path),
    )

    try:
        await env.reset()
        if incremental_embeddings:
            await _embed_all_bios(
                db_path=db_path,
                model=embedding_model,
                batch_size=embedding_batch_size,
            )

        last_post_id = _get_max_id(db_path, "post", "post_id")
        await _seed_posts(env, seeding_data, seed_post_count)
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
        )
        _snapshot_database(
            db_path=db_path,
            snapshot_dir=step_snapshot_dir,
            label="after_seeding",
        )

        for round_num in range(llm_rounds):
            await _llm_round(
                env,
                agent_action_ratio,
                recommender,
                round_num,
                incremental_embeddings=incremental_embeddings,
                db_path=db_path,
                embedding_model=embedding_model,
                embedding_batch_size=embedding_batch_size,
            )
            _capture_step_metrics(
                db_path=db_path,
                metrics_path=step_metrics_path,
                label=f"round_{round_num + 1}",
                round_num=round_num + 1,
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
    )
    _snapshot_database(
        db_path=db_path,
        snapshot_dir=step_snapshot_dir,
        label="final",
    )


async def _seed_posts(env, seeding_data: list[dict], seed_post_count: int) -> None:
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
        await env.step(actions)
        
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
                _prune_agent_memory(agent, keep_last=10)
            except Exception:
                pass  # Ignore if memory clearing fails
    
    prev_post_id = _get_max_id(db_path, "post", "post_id") if incremental_embeddings and db_path else 0
    prev_comment_id = _get_max_id(db_path, "comment", "comment_id") if incremental_embeddings and db_path else 0

    actions = {
        agent: LLMAction()
        for _, agent in selected_agents
    }

    if actions:
        await env.step(actions)
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
    seed_post_count: int = 20,
    llm_rounds: int = 1,
    agent_action_ratio: float = 0.3,
    recommendation_type: RecommendationType | str = RecommendationType.RANDOM,
    model_type: str | ModelType = "gpt-4o",
    model_temperature: float | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_batch_size: int = 32,
    skip_embeddings: bool = False,
    step_metrics_path: Path | str | None = None,
    step_snapshot_dir: Path | str | None = None,
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
        )
    )
