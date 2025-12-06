import argparse
import csv
import subprocess
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict

import yaml

from src.simulation.simulation import run


RESULTS_DIR = Path("results")
LATEST_RUN_FILE = RESULTS_DIR / "latest_run.txt"
INDEX_FILE = RESULTS_DIR / "index.csv"
CONFIG_DIR_NAME = "config"
CONFIG_SNAPSHOT_FILES = (
    Path(".env"),
    Path(".mise.toml"),
    Path("docker-compose.yml"),
    Path("pyproject.toml"),
    Path("uv.lock"),
)


def _ensure_config_dir(run_dir: Path) -> Path:
    config_dir = run_dir / CONFIG_DIR_NAME
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def snapshot_config_files(run_dir: Path) -> list[str]:
    """Copy selected project configuration files into the run directory."""

    copied: list[str] = []
    config_dir = _ensure_config_dir(run_dir)

    for source in CONFIG_SNAPSHOT_FILES:
        if source.exists():
            destination = config_dir / source.name
            shutil.copy2(source, destination)
            copied.append(str(destination.relative_to(run_dir)))

    return copied


def write_run_arguments(run_dir: Path, args: argparse.Namespace) -> Path:
    """Persist the parsed CLI arguments for reproducibility."""

    config_dir = _ensure_config_dir(run_dir)
    args_path = config_dir / "run_args.yaml"
    normalized = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    with args_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(normalized, fh, allow_unicode=True, sort_keys=True)
    return args_path


def write_neo4j_config(run_dir: Path) -> Path:
    """Write connection defaults for Neo4j exports."""

    config_dir = _ensure_config_dir(run_dir)
    neo4j_path = config_dir / "neo4j.yaml"
    payload = {
        "uri": "bolt://localhost:7687",
        "user": "neo4j",
        "password_env_var": "NEO4J_PASSWORD",
        "default_password": "neo4j1234",
        "docker_compose": "config/docker-compose.yml",
    }
    with neo4j_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, allow_unicode=True, sort_keys=True)
    return neo4j_path


def get_git_metadata() -> Dict[str, Any]:
    """Collect lightweight git information for the run metadata."""

    git_info: Dict[str, Any] = {}

    try:
        root_check = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return git_info

    if root_check.stdout.strip().lower() != "true":
        return git_info

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
        git_info["commit"] = commit.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True)
        git_info["branch"] = branch.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
        git_info["is_dirty"] = bool(status.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return git_info


def create_run_directory(base_dir: Path, run_name: str | None) -> tuple[Path, str]:
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
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
    fieldnames = [
        "run_dir",
        "timestamp",
        "persona_path",
        "persona_copy",
        "seed_post_count",
        "llm_rounds",
        "database_path",
        "note",
        "status",
        "error",
        "run_args",
        "config_snapshot",
    ]

    existing_rows: list[Dict[str, Any]] = []
    if INDEX_FILE.exists():
        with INDEX_FILE.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                normalized = {name: row.get(name) for name in fieldnames}
                existing_rows.append(normalized)

    normalized_entry = {name: entry.get(name) for name in fieldnames}
    existing_rows.append(normalized_entry)

    with INDEX_FILE.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)


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
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    else:
        base_dir = Path(args.output_dir)
        run_dir, timestamp = create_run_directory(base_dir, args.run_name)
        database_path = run_dir / "simulation.db"

    config_snapshot = snapshot_config_files(run_dir)
    args_path = write_run_arguments(run_dir, args)
    neo4j_config_path = write_neo4j_config(run_dir)

    persona_copy = None
    if not args.no_persona_copy:
        destination = run_dir / persona_path.name
        shutil.copy2(persona_path, destination)
        persona_copy = destination

    git_info = get_git_metadata()

    status = "success"
    error_message: str | None = None
    run_error: Exception | None = None

    try:
        run(
            persona_path=persona_path,
            database_path=database_path,
            seed_post_count=args.seed_post_count,
            llm_rounds=args.llm_rounds,
        )
    except Exception as exc:  # noqa: BLE001 - capture all errors for metadata recording
        status = "failed"
        error_message = f"{exc.__class__.__name__}: {exc}"
        run_error = exc
    finally:
        resolved_run_dir = run_dir.resolve()
        resolved_database = Path(database_path).resolve()
        metadata: Dict[str, Any] = {
            "run_dir": str(resolved_run_dir),
            "timestamp": timestamp,
            "persona_path": str(persona_path.resolve()),
            "persona_copy": str(persona_copy.resolve()) if persona_copy else None,
            "seed_post_count": args.seed_post_count,
            "llm_rounds": args.llm_rounds,
            "database_path": str(resolved_database),
            "note": args.note or "",
            "status": status,
            "error": error_message,
            "config_snapshot": config_snapshot,
            "run_args_file": str(args_path.relative_to(run_dir)),
            "neo4j_config_file": str(neo4j_config_path.relative_to(run_dir)),
            "git": git_info,
        }

        write_metadata(run_dir, metadata)

        index_entry = {
            "run_dir": metadata["run_dir"],
            "timestamp": metadata["timestamp"],
            "persona_path": metadata["persona_path"],
            "persona_copy": metadata["persona_copy"],
            "seed_post_count": metadata["seed_post_count"],
            "llm_rounds": metadata["llm_rounds"],
            "database_path": metadata["database_path"],
            "note": metadata["note"],
            "status": metadata["status"],
            "error": metadata["error"],
            "run_args": metadata["run_args_file"],
            "config_snapshot": ";".join(config_snapshot),
        }
        update_index(index_entry)

        if status == "success":
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            LATEST_RUN_FILE.write_text(str(resolved_run_dir), encoding="utf-8")
            print(f"Run stored in: {run_dir}")
        else:
            print(f"Run failed; artifacts captured in: {run_dir}")

    if run_error is not None:
        raise run_error


if __name__ == "__main__":
    main()
