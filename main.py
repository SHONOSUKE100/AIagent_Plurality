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
        import argparse
        import csv
        import shutil
        from datetime import datetime
        from pathlib import Path
        from typing import Dict, Any

        import yaml

        from src.simulation.simulation import run


        RESULTS_DIR = Path("results")
        LATEST_RUN_FILE = RESULTS_DIR / "latest_run.txt"
        INDEX_FILE = RESULTS_DIR / "index.csv"


        def create_run_directory(base_dir: Path, run_name: str | None) -> tuple[Path, str]:
            base_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            default_name = run_name or timestamp
            candidate = base_dir / default_name
            suffix = 1
            while candidate.exists():
                candidate = base_dir / f"{default_name}-{suffix:02d}"
                suffix += 1
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate, timestamp


        def write_metadata(run_dir: Path, data: Dict[str, Any]) -> None:
            metadata_path = run_dir / "metadata.yaml"
            with metadata_path.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)


        def update_index(entry: Dict[str, Any]) -> None:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            is_new = not INDEX_FILE.exists()
            fieldnames = [
                "run_dir",
                "timestamp",
                "persona_path",
                "persona_copy",
                "seed_post_count",
                "llm_rounds",
                "database_path",
                "note",
            ]
            with INDEX_FILE.open("a", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                if is_new:
                    writer.writeheader()
                writer.writerow(entry)


        def main() -> None:
            parser = argparse.ArgumentParser(description="Run the AI agent plurality simulation.")
            parser.add_argument(
                "--persona-path",
                default="data/persona/persona.json",
                help="Path to the persona JSON file.",
            )
            parser.add_argument(
                "--database-path",
                help="Optional explicit SQLite path. If omitted, a results/<run>/simulation.db file is created.",
            )
            parser.add_argument(
                "--output-dir",
                default="results",
                help="Base directory for storing experiment runs.",
            )
            parser.add_argument(
                "--run-name",
                help="Optional run name. Defaults to timestamp.",
            )
            parser.add_argument(
                "--note",
                help="Optional note describing the run (stored in metadata).",
            )
            parser.add_argument(
                "--no-persona-copy",
                action="store_true",
                help="Do not copy the persona file into the run directory.",
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

            persona_path = Path(args.persona_path)
            if not persona_path.exists():
                raise FileNotFoundError(f"Persona file not found: {persona_path}")

            if args.database_path:
                database_path = Path(args.database_path)
                run_dir = database_path.parent
                run_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            else:
                base_dir = Path(args.output_dir)
                run_dir, timestamp = create_run_directory(base_dir, args.run_name)
                database_path = run_dir / "simulation.db"

            run(
                persona_path=persona_path,
                database_path=database_path,
                seed_post_count=args.seed_post_count,
                llm_rounds=args.llm_rounds,
            )

            persona_copy = None
            if not args.no_persona_copy:
                destination = run_dir / persona_path.name
                shutil.copy2(persona_path, destination)
                persona_copy = destination

            metadata = {
                "run_dir": str(run_dir.resolve()),
                "timestamp": timestamp,
                "persona_path": str(persona_path.resolve()),
                "persona_copy": str(persona_copy.resolve()) if persona_copy else None,
                "seed_post_count": args.seed_post_count,
                "llm_rounds": args.llm_rounds,
                "database_path": str(database_path.resolve()),
                "note": args.note or "",
            }
            write_metadata(run_dir, metadata)

            update_index(metadata)

            LATEST_RUN_FILE.write_text(str(run_dir.resolve()), encoding="utf-8")

            print(f"Run stored in: {run_dir}")
