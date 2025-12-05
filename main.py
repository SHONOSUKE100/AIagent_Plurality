import argparse
from pathlib import Path

from src.simulation.simulation import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AI agent plurality simulation.")
    parser.add_argument(
        "--persona-path",
        default="data/persona/persona.json",
        help="Path to the persona JSON file.",
    )
    parser.add_argument(
        "--database-path",
        default="data/twitter_simulation.db",
        help="SQLite database destination for the simulation run.",
    )
    parser.add_argument(
        "--seed-post-count",
        type=int,
        default=20,
        help="Number of agents that create initial posts before the LLM rounds.",
    )
    parser.add_argument(
        "--llm-rounds",
        type=int,
        default=1,
        help="How many rounds of LLM-driven actions to execute after seeding posts.",
    )

    args = parser.parse_args()

    run(
        persona_path=Path(args.persona_path),
        database_path=Path(args.database_path),
        seed_post_count=args.seed_post_count,
        llm_rounds=args.llm_rounds,
    )


if __name__ == "__main__":
    main()
