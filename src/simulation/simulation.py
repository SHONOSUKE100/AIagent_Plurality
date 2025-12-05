"""Simulation entry points for running the OASIS environment."""

from __future__ import annotations

import asyncio
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


async def run_simulation(
    *,
    persona_path: Path | str,
    database_path: Path | str,
    seed_post_count: int = 20,
    llm_rounds: int = 1,
    platform: DefaultPlatformType = DefaultPlatformType.TWITTER,
    available_actions: Sequence[ActionType] | None = None,
) -> None:
    """Run the social simulation once using personas from ``persona_path``."""

    personas = load_personas(persona_path)
    model = create_default_model()
    agent_graph = build_agent_graph(personas, model, available_actions=available_actions)

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
        await _seed_posts(env, seed_post_count)
        for _ in range(llm_rounds):
            await _llm_round(env)
    finally:
        await env.close()


async def _seed_posts(env, seed_post_count: int) -> None:
    agent_items = list(env.agent_graph.get_agents())
    if not agent_items:
        return

    sample_count = min(seed_post_count, len(agent_items))
    selected = random.sample(agent_items, sample_count)

    actions = {}
    for agent_id, agent in selected:
        user_info = getattr(agent, "user_info", None)
        display_name = getattr(user_info, "name", f"Agent {agent_id}")
        actions[agent] = [
            ManualAction(
                action_type=ActionType.CREATE_POST,
                action_args={
                    "content": f"{display_name} is joining the simulation!",
                },
            )
        ]

    if actions:
        await env.step(actions)


async def _llm_round(env) -> None:
    actions = {
        agent: LLMAction()
        for _, agent in env.agent_graph.get_agents()
    }

    if actions:
        await env.step(actions)


def run(
    persona_path: Path | str = Path("data/persona/persona.json"),
    database_path: Path | str = Path("data/twitter_simulation.db"),
    seed_post_count: int = 20,
    llm_rounds: int = 1,
) -> None:
    """Convenience wrapper that mirrors the notebook execution."""

    asyncio.run(
        run_simulation(
            persona_path=persona_path,
            database_path=database_path,
            seed_post_count=seed_post_count,
            llm_rounds=llm_rounds,
            available_actions=DEFAULT_AVAILABLE_ACTIONS,
        )
    )