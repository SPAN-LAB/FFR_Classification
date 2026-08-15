"""Generate a CHTC analysis manifest and optionally submit it."""

from __future__ import annotations

import argparse
import subprocess
from itertools import product
from pathlib import Path

from src.analysis.settings import ANALYSIS_TYPES, MODEL_TRAINING_OPTIONS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit configured models for every subject and analysis type."
    )
    parser.add_argument(
        "models",
        nargs="*",
        default=["FFNN"],
        help="Model names, or 'all' for every model with configured training options",
    )
    parser.add_argument(
        "--analyses",
        nargs="+",
        choices=ANALYSIS_TYPES,
        default=list(ANALYSIS_TYPES),
    )
    parser.add_argument("--jobs-dir", type=Path, default=Path.cwd())
    parser.add_argument("--subjects-file", default="subjects.txt")
    parser.add_argument("--manifest", default="analysis_jobs.txt")
    parser.add_argument("--submit-file", default="analysis.sub")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write and display the manifest without calling condor_submit",
    )
    return parser


def _read_subjects(path: Path) -> list[str]:
    subjects = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not subjects:
        raise ValueError(f"No subjects found in {path}")
    return subjects


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_manifest(
    *,
    models: list[str],
    subjects: list[str],
    analyses: list[str],
    git_commit: str,
) -> list[str]:
    models = expand_models(models)
    return [
        f"{model} {subject} {analysis} {git_commit}"
        for model, subject, analysis in product(models, subjects, analyses)
    ]


def expand_models(models: list[str]) -> list[str]:
    if [model.lower() for model in models] == ["all"]:
        return list(MODEL_TRAINING_OPTIONS)
    if not models:
        raise ValueError("At least one model is required")
    return models


def main() -> None:
    args = _parser().parse_args()
    jobs_dir = args.jobs_dir.resolve()
    subjects_path = jobs_dir / args.subjects_file
    submit_path = jobs_dir / args.submit_file
    if not submit_path.is_file():
        raise FileNotFoundError(f"Submit file not found: {submit_path}")

    models = expand_models(args.models)
    rows = build_manifest(
        models=models,
        subjects=_read_subjects(subjects_path),
        analyses=args.analyses,
        git_commit=_git_commit(),
    )
    manifest_path = jobs_dir / args.manifest
    manifest_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    (jobs_dir / "logs").mkdir(parents=True, exist_ok=True)
    for analysis, model in product(args.analyses, models):
        (jobs_dir / "analyses" / analysis / model).mkdir(
            parents=True,
            exist_ok=True,
        )

    print(f"Prepared {len(rows)} jobs in {manifest_path}")
    for row in rows[:10]:
        print(f"  {row}")
    if len(rows) > 10:
        print(f"  ... and {len(rows) - 10} more")

    if args.dry_run:
        return
    subprocess.run(
        ["condor_submit", args.submit_file],
        cwd=jobs_dir,
        check=True,
    )


if __name__ == "__main__":
    main()
