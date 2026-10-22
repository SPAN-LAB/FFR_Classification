"""Pre-trained checkpoints for transfer analysis, and their files on disk."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np


CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_WEIGHTS_SUFFIX = ".pretrained.pt"
CHECKPOINT_METADATA_SUFFIX = ".pretrained.json"

# A loaded checkpoint must match the current run on every one of these fields.
CHECKPOINT_MATCH_FIELDS = (
    "model",
    "held_out_subject",
    "pretraining_subjects",
    "pretraining_label_counts",
    "labels_map",
    "n_classes",
    "n_times",
    "pretraining_options",
    "subaverage_size",
    "trim_start_ms",
    "trim_end_ms",
    "data_var",
)


@dataclass
class PretrainedCheckpoint:
    status: str  # "success" or "failure"
    source: str  # "trained" or "loaded"
    state_dict: dict[str, Any] | None  # CPU tensors; None unless status == "success"
    metadata: dict[str, Any]
    source_path: str | None = None
    source_git_commit: str | None = None
    error_traceback: str | None = None


class CheckpointError(Exception):
    """A checkpoint file is missing, empty, failed, or unreadable."""


def jsonable(value: Any) -> Any:
    """Normalize metadata so values compare equal before and after a JSON round trip."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return str(value)


def checkpoint_paths(path: str | Path) -> tuple[Path, Path]:
    """Return the weights and metadata paths for a .pt, a .json, or their prefix."""
    text = str(path)
    for suffix in (CHECKPOINT_WEIGHTS_SUFFIX, CHECKPOINT_METADATA_SUFFIX):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return (
        Path(f"{text}{CHECKPOINT_WEIGHTS_SUFFIX}"),
        Path(f"{text}{CHECKPOINT_METADATA_SUFFIX}"),
    )


def _atomic_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid4().hex}.tmp")


def write_checkpoint(
    checkpoint: PretrainedCheckpoint,
    output_prefix: str | Path,
    *,
    git_commit: str | None,
) -> tuple[Path, Path]:
    """Always write both files, atomically. On success, the .pt holds the state
    dict; otherwise it is empty and the JSON records the status and traceback."""
    import torch

    weights_path, metadata_path = checkpoint_paths(output_prefix)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    has_weights = checkpoint.status == "success" and checkpoint.state_dict is not None

    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "status": checkpoint.status if has_weights else "failure",
        "source": checkpoint.source,
        "source_path": checkpoint.source_path,
        # The commit that trained the weights; a reloaded checkpoint keeps its own.
        "source_git_commit": (
            checkpoint.source_git_commit
            if checkpoint.source == "loaded"
            else git_commit
        ),
        "written_by_git_commit": git_commit,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": jsonable(checkpoint.metadata),
        "traceback": checkpoint.error_traceback,
    }

    temporary_weights = _atomic_path(weights_path)
    temporary_metadata = _atomic_path(metadata_path)
    try:
        if has_weights:
            torch.save(checkpoint.state_dict, temporary_weights)
        else:
            temporary_weights.write_bytes(b"")
        with temporary_metadata.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
            file.write("\n")
        os.replace(temporary_weights, weights_path)
        os.replace(temporary_metadata, metadata_path)
    finally:
        temporary_weights.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)

    return weights_path, metadata_path


def read_checkpoint(path: str | Path) -> PretrainedCheckpoint:
    """Accept the .pt, the .json, or their shared prefix. Raise CheckpointError
    if either file is missing, the .pt is empty, or the JSON status is not
    "success"."""
    import torch

    weights_path, metadata_path = checkpoint_paths(path)
    if not metadata_path.is_file():
        raise CheckpointError(f"Checkpoint metadata not found: {metadata_path}")
    if not weights_path.is_file():
        raise CheckpointError(f"Checkpoint weights not found: {weights_path}")

    try:
        with metadata_path.open(encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointError(f"Cannot read checkpoint metadata {metadata_path}: {error}") from error

    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError(
            f"Unsupported checkpoint schema {payload.get('schema_version')!r} "
            f"in {metadata_path}."
        )
    if payload.get("status") != "success":
        raise CheckpointError(
            f"Checkpoint {metadata_path} has status {payload.get('status')!r}; "
            "only successful checkpoints can be loaded."
        )
    if weights_path.stat().st_size == 0:
        raise CheckpointError(f"Checkpoint weights file is empty: {weights_path}")

    try:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise CheckpointError(f"Cannot load checkpoint weights {weights_path}: {error}") from error

    return PretrainedCheckpoint(
        status="success",
        source="loaded",
        state_dict=state_dict,
        metadata=payload.get("metadata") or {},
        source_path=str(weights_path),
        source_git_commit=payload.get("source_git_commit"),
    )
