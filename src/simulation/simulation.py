"""Simulation entry points for running the OASIS environment."""

from __future__ import annotations

import asyncio
import json
import os
import random
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
from ..algorithms import RecommendationType, create_recommender


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
            Options: "random", "collaborative", "bridging", "diversity", "echo_chamber", "hybrid"
        platform: The platform type for the simulation.
        available_actions: Available action types for agents.
    """

    personas = load_personas(persona_path)
    model = create_default_model()
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
        await _seed_posts(env, seeding_data, seed_post_count)
        for round_num in range(llm_rounds):
            await _llm_round(env, agent_action_ratio, recommender, round_num)
    finally:
        await env.close()


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
    recommender = None,
    round_num: int = 0,
    clear_memory_before_action: bool = True,
) -> None:
    """Execute one LLM round with a random subset of agents.
    
    Args:
        env: The simulation environment.
        agent_action_ratio: Ratio of agents (0.0-1.0) to randomly select for this round.
        recommender: Optional custom recommender for content moderation.
        round_num: Current round number (for logging/debugging).
        clear_memory_before_action: Whether to clear agent memory before LLM action
            to prevent tool_calls state corruption.
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
                agent.memory.clear()
            except Exception:
                pass  # Ignore if memory clearing fails
    
    actions = {
        agent: LLMAction()
        for _, agent in selected_agents
    }

    if actions:
        await env.step(actions)


def run(
    persona_path: Path | str = Path("data/persona/persona.json"),
    database_path: Path | str = Path("data/twitter_simulation.db"),
    seeding_path: Path | str = Path("data/seeding.json"),
    seed_post_count: int = 20,
    llm_rounds: int = 1,
    agent_action_ratio: float = 0.3,
    recommendation_type: RecommendationType | str = RecommendationType.RANDOM,
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
            Options: "random", "collaborative", "bridging", "diversity", "echo_chamber", "hybrid"
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
        )
    )
