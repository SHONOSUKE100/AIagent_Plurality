import argparse
import csv
import subprocess
import shutil
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict

import yaml

from src.simulation.simulation import run
from src.simulation.embedding_store import DEFAULT_EMBEDDING_MODEL


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


def write_run_arguments(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    filename: str = "run_args.yaml",
) -> Path:
    """Persist the parsed CLI arguments for reproducibility."""

    config_dir = _ensure_config_dir(run_dir)
    args_path = config_dir / filename
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
    default_name = f"{run_name}-{timestamp}" if run_name else timestamp
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
        "warmup_random_rounds",
        "database_path",
        "embedding_model",
        "embedding_batch_size",
        "skip_embeddings",
        "note",
        "status",
        "error",
        "run_args",
        "config_snapshot",
        "step_metrics_file",
        "step_snapshot_dir",
        "resume_run_dir",
        "resumed_from_round",
        "resumed_from_snapshot",
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


def _detect_last_round_from_snapshots(snapshot_dir: Path) -> int | None:
    if not snapshot_dir.exists():
        return None

    max_round: int | None = None
    for path in snapshot_dir.glob("round_*.db"):
        match = re.fullmatch(r"round_(\d+)\.db", path.name)
        if not match:
            continue
        value = int(match.group(1))
        max_round = value if max_round is None else max(max_round, value)

    if max_round is not None:
        return max_round
    if (snapshot_dir / "after_seeding.db").exists():
        return 0
    return None


def _detect_last_round_from_metrics(metrics_path: Path) -> int | None:
    if not metrics_path.exists():
        return None

    max_round: int | None = None
    with metrics_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            value = row.get("round")
            if value is None or value == "":
                continue
            try:
                round_num = int(float(value))
            except ValueError:
                continue
            max_round = round_num if max_round is None else max(max_round, round_num)
    return max_round


def _infer_round_from_snapshot(snapshot_path: Path) -> int | None:
    if snapshot_path.name == "after_seeding.db":
        return 0
    match = re.fullmatch(r"round_(\d+)\.db", snapshot_path.name)
    if match:
        return int(match.group(1))
    return None


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
        "--resume-run-dir",
        help="Resume an interrupted run from an existing results/<run> directory.",
    )
    parser.add_argument(
        "--resume-from-round",
        type=int,
        help="Last fully completed round number to resume from (overrides auto-detect).",
    )
    parser.add_argument(
        "--resume-from-snapshot",
        help="Path to a snapshot DB (e.g., step_snapshots/round_47.db) to resume from.",
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
        default=10,
        help="Number of agents that create initial posts before the LLM rounds.",
    )
    parser.add_argument(
        "--llm-rounds",
        type=int,
        default=1,
        help="How many rounds of LLM-driven actions to execute after seeding posts.",
    )
    parser.add_argument(
        "--seeding-path",
        default="data/seeding.json",
        help="Path to the seeding JSON file containing initial post contents.",
    )
    parser.add_argument(
        "--agent-action-ratio",
        type=float,
        default=0.2,
        help="Ratio of agents (0.0-1.0) to randomly select for each LLM round. Default is 0.2 (20%%).",
    )
    parser.add_argument(
        "--recommendation-type",
        default="random",
        choices=["random", "collaborative", "bridging"],
        help="Type of recommendation algorithm to use for content moderation.",
    )
    parser.add_argument(
        "--model-type",
        default="gpt-5-nano-2025-08-07",
        help="Model identifier to use (e.g., gpt-4o, gpt-4o-mini, gpt-5-nano).",
    )
    parser.add_argument(
        "--model-temperature",
        type=float,
        default=None,
        help="Optional temperature override. Leave unset to use model defaults (required for some models like gpt-5-nano).",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model used for posts/comments/bios. Set to empty to skip embedding.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=16,
        help="Batch size for embedding API calls.",
    )
    parser.add_argument(
        "--max-memory-messages",
        type=int,
        default=5,
        help="Number of past messages to retain in agent memory to reduce prompt and RAM size.",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip the embedding step to avoid extra API usage.",
    )
    parser.add_argument(
        "--warmup-random-rounds",
        type=int,
        default=0,
        help="Use random recommender for the first N rounds, then switch to the chosen recommender (exploration warm-up).",
    )
    parser.add_argument(
        "--rate-limit-initial-delay-sec",
        type=int,
        default=600,
        help="Initial backoff when rate limited (seconds). Default 600 (10 minutes).",
    )
    parser.add_argument(
        "--rate-limit-retry-delay-sec",
        type=int,
        default=300,
        help="Backoff between subsequent rate limit retries (seconds). Default 300 (5 minutes).",
    )
    parser.add_argument(
        "--max-rate-limit-retries",
        type=int,
        default=50,
        help="Maximum rate limit retries before failing the run.",
    )

    args = parser.parse_args()

    persona_path = Path(args.persona_path)
    if not persona_path.exists():
        raise FileNotFoundError(f"Persona file not found: {persona_path}")

    resume_run_dir = Path(args.resume_run_dir) if args.resume_run_dir else None
    resume_snapshot_path: Path | None = (
        Path(args.resume_from_snapshot) if args.resume_from_snapshot else None
    )
    resume_last_round: int | None = args.resume_from_round

    if resume_run_dir:
        if not resume_run_dir.exists():
            raise FileNotFoundError(f"resume-run-dir not found: {resume_run_dir}")
        run_dir = resume_run_dir
        if args.database_path:
            candidate = Path(args.database_path)
            if run_dir not in candidate.parents and candidate.parent != run_dir:
                raise ValueError(
                    "When using --resume-run-dir, --database-path must be inside that directory."
                )
            database_path = candidate
        else:
            database_path = run_dir / "simulation.db"
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    else:
        if args.database_path:
            database_path = Path(args.database_path)
            run_dir = database_path.parent
            run_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        else:
            base_dir = Path(args.output_dir)
            run_dir, timestamp = create_run_directory(base_dir, args.run_name)
            database_path = run_dir / "simulation.db"

    if resume_run_dir:
        snapshot_dir = run_dir / "step_snapshots"
        metrics_path = run_dir / "step_metrics.csv"
        detected_from_snapshots = _detect_last_round_from_snapshots(snapshot_dir)
        detected_from_metrics = _detect_last_round_from_metrics(metrics_path)

        if resume_snapshot_path is None:
            if detected_from_snapshots is not None:
                if detected_from_snapshots > 0:
                    resume_snapshot_path = snapshot_dir / f"round_{detected_from_snapshots}.db"
                else:
                    candidate = snapshot_dir / "after_seeding.db"
                    resume_snapshot_path = candidate if candidate.exists() else None
            if resume_snapshot_path is None and database_path.exists():
                resume_snapshot_path = database_path

        if resume_snapshot_path is None or not resume_snapshot_path.exists():
            raise FileNotFoundError(
                "Could not find a resume snapshot. Specify --resume-from-snapshot explicitly."
            )

        if resume_last_round is None:
            resume_last_round = _infer_round_from_snapshot(resume_snapshot_path)
        if resume_last_round is None:
            resume_last_round = detected_from_snapshots
        if resume_last_round is None:
            resume_last_round = detected_from_metrics
        if resume_last_round is None:
            resume_last_round = 0

        if resume_last_round < 0:
            raise ValueError("--resume-from-round must be non-negative.")
        if resume_last_round > args.llm_rounds:
            raise ValueError(
                f"resume-from-round ({resume_last_round}) exceeds llm-rounds ({args.llm_rounds})."
            )

        print(
            f"[resume] run_dir={run_dir} snapshot={resume_snapshot_path.name} last_round={resume_last_round}"
        )

    config_snapshot = snapshot_config_files(run_dir)
    run_args_filename = (
        f"run_args_resume_{timestamp}.yaml" if resume_run_dir else "run_args.yaml"
    )
    args_path = write_run_arguments(run_dir, args, filename=run_args_filename)
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


    # Embeddingモデルとスキップ判定のロジックを明確化
    if args.skip_embeddings or not args.embedding_model:
        skip_embeddings = True
        effective_embedding_model = None  # 埋め込み自体をスキップ
    else:
        skip_embeddings = False
        effective_embedding_model = args.embedding_model  # 明示指定のみ利用

    try:
        run(
            persona_path=persona_path,
            database_path=database_path,
            seeding_path=Path(args.seeding_path),
            seed_post_count=args.seed_post_count,
            llm_rounds=args.llm_rounds,
            agent_action_ratio=args.agent_action_ratio,
            recommendation_type=args.recommendation_type,
            model_type=args.model_type,
            model_temperature=args.model_temperature,
            embedding_model=effective_embedding_model or "",
            embedding_batch_size=args.embedding_batch_size,
            skip_embeddings=skip_embeddings,
            step_metrics_path=run_dir / "step_metrics.csv",
            step_snapshot_dir=run_dir / "step_snapshots",
            max_memory_messages=args.max_memory_messages,
            warmup_random_rounds=args.warmup_random_rounds,
            resume_from_snapshot=resume_snapshot_path,
            resume_last_completed_round=resume_last_round,
            rate_limit_initial_delay_sec=args.rate_limit_initial_delay_sec,
            rate_limit_retry_delay_sec=args.rate_limit_retry_delay_sec,
            max_rate_limit_retries=args.max_rate_limit_retries,
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
            "seeding_path": str(Path(args.seeding_path).resolve()),
            "seed_post_count": args.seed_post_count,
            "llm_rounds": args.llm_rounds,
            "warmup_random_rounds": args.warmup_random_rounds,
            "agent_action_ratio": args.agent_action_ratio,
            "recommendation_type": args.recommendation_type,
            "database_path": str(resolved_database),
            # skip_embeddingsがTrueならembedding_modelはNoneまたは空文字列
            "embedding_model": effective_embedding_model if not skip_embeddings else None,
            "embedding_batch_size": args.embedding_batch_size,
            "skip_embeddings": skip_embeddings,
            "note": args.note or "",
            "status": status,
            "error": error_message,
            "config_snapshot": config_snapshot,
            "run_args_file": str(args_path.relative_to(run_dir)),
            "neo4j_config_file": str(neo4j_config_path.relative_to(run_dir)),
            "step_metrics_file": "step_metrics.csv",
            "step_snapshot_dir": "step_snapshots",
            "resume_run_dir": str(resume_run_dir.resolve()) if resume_run_dir else None,
            "resumed_from_round": resume_last_round,
            "resumed_from_snapshot": str(resume_snapshot_path.resolve()) if resume_snapshot_path else None,
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
            "warmup_random_rounds": metadata["warmup_random_rounds"],
            "database_path": metadata["database_path"],
            # skip_embeddingsがTrueならembedding_modelはNoneまたは空文字列
            "embedding_model": metadata["embedding_model"],
            "embedding_batch_size": metadata["embedding_batch_size"],
            "skip_embeddings": metadata["skip_embeddings"],
            "note": metadata["note"],
            "status": metadata["status"],
            "error": metadata["error"],
            "run_args": metadata["run_args_file"],
            "config_snapshot": ";".join(config_snapshot),
            "step_metrics_file": metadata.get("step_metrics_file"),
            "step_snapshot_dir": metadata.get("step_snapshot_dir"),
            "resume_run_dir": metadata.get("resume_run_dir"),
            "resumed_from_round": metadata.get("resumed_from_round"),
            "resumed_from_snapshot": metadata.get("resumed_from_snapshot"),
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
