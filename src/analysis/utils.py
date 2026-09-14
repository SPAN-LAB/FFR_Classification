"""Utilities for discovering data and reading analysis results."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

from ..core import EEGSubject, EEGTrial


def get_mats(folder_path: str) -> list[str]:
    """Return sorted MATLAB file paths directly inside a folder."""
    if not os.path.isdir(folder_path):
        raise ValueError("The path provided is not a folder path.")
    return sorted(
        os.path.join(folder_path, filename)
        for filename in os.listdir(folder_path)
        if filename.endswith(".mat")
    )


def get_trailing_number(value: str) -> int | None:
    stem = Path(value).stem
    digits = []
    for character in reversed(stem):
        if not character.isdigit():
            break
        digits.append(character)
    return int("".join(reversed(digits))) if digits else None


def get_result_records(dir_path: str) -> list[dict]:
    """Read structured summaries and legacy pickled analysis results."""
    directory = Path(dir_path)
    records = []

    for filename in sorted(directory.rglob("*.summary.json")):
        with filename.open("r", encoding="utf-8") as file:
            summary = json.load(file)
        common = {
            "model": summary.get("model", "unknown"),
            "subject": summary.get("subject", filename.parent.name),
            "analysis": summary.get("analysis", "unknown"),
        }
        conditions = summary.get("conditions")
        if conditions is None:
            conditions = [
                {
                    "value": summary.get("value"),
                    "status": summary.get("status"),
                    "metrics": summary.get("metrics", {}),
                }
            ]
        for condition in conditions:
            accuracy = condition.get("metrics", {}).get("accuracy")
            value = condition.get("value")
            if accuracy is None or value is None:
                continue
            records.append(
                {
                    **common,
                    "value": int(value),
                    "accuracy": float(accuracy),
                    "status": condition.get("status", "unknown"),
                }
            )

    for filename in sorted(directory.rglob("*.pkl")):
        with filename.open("rb") as file:
            subject = pickle.load(file)
        if not isinstance(subject, EEGSubject):
            continue
        value = get_trailing_number(str(filename))
        if value is None:
            continue
        records.append(
            {
                "model": filename.parent.parent.name,
                "subject": subject.name,
                "analysis": "legacy",
                "value": value,
                "accuracy": EEGTrial.get_accuracy(subject.trials),
                "status": "success",
            }
        )

    return sorted(
        records,
        key=lambda item: (item["model"], item["subject"], item["value"]),
    )


def get_results(dir_path: str) -> list[tuple[int, float]]:
    """Return value/accuracy pairs for plotting compatibility."""
    return [
        (record["value"], record["accuracy"])
        for record in get_result_records(dir_path)
    ]
