from __future__ import annotations

import csv
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from ..core import AnalysisPipeline


PREDICTION_FIELDS = [
    "subject",
    "model",
    "analysis",
    "value",
    "fold",
    "trial_index",
    "true_label_json",
    "predicted_label_json",
    "prediction_distribution_json",
    "correct",
]


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)


def _json_cell(value: Any) -> str:
    return json.dumps(_json_value(value), ensure_ascii=True, sort_keys=True)


def _prediction_rows(
    pipeline: AnalysisPipeline | None,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    if pipeline is None:
        return []

    rows = []
    for subject in pipeline.subjects:
        folds = subject.folds if subject.folds else [subject.trials]
        for fold_index, trials in enumerate(folds):
            for trial in trials:
                distribution = [
                    {
                        "label": _json_value(label),
                        "probability": float(probability),
                    }
                    for label, probability in (
                        trial.prediction_distribution or {}
                    ).items()
                ]
                rows.append(
                    {
                        "subject": subject.name,
                        "model": metadata["model"],
                        "analysis": metadata["analysis"],
                        "value": metadata["value"],
                        "fold": fold_index,
                        "trial_index": trial.trial_index,
                        "true_label_json": _json_cell(trial.label),
                        "predicted_label_json": _json_cell(trial.prediction),
                        "prediction_distribution_json": _json_cell(distribution),
                        "correct": int(trial.prediction == trial.label),
                    }
                )
    return rows


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "accuracy": None,
            "num_predictions": 0,
            "per_label": [],
        }

    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for row in rows:
        label_json = row["true_label_json"]
        item = grouped.setdefault(
            label_json,
            {
                "label": json.loads(label_json),
                "correct": 0,
                "total": 0,
            },
        )
        item["correct"] += row["correct"]
        item["total"] += 1

    per_label = []
    for item in grouped.values():
        per_label.append(
            {
                **item,
                "accuracy": item["correct"] / item["total"],
            }
        )

    return {
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "num_predictions": len(rows),
        "per_label": per_label,
    }


def _atomic_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid4().hex}.tmp")


def write_result_files(
    output_prefix: str | Path,
    *,
    pipeline: AnalysisPipeline | None,
    metadata: dict[str, Any],
    error_traceback: str | None = None,
) -> tuple[Path, Path]:
    """Write JSON metrics and trial-level CSV predictions for one cluster job."""
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(f"{output_prefix}.summary.json")
    predictions_path = Path(f"{output_prefix}.predictions.csv")
    rows = _prediction_rows(pipeline, metadata)
    summary = {
        "schema_version": 1,
        **metadata,
        "metrics": _metrics(rows),
    }
    if error_traceback is not None:
        summary["traceback"] = error_traceback

    temporary_summary = _atomic_path(summary_path)
    temporary_predictions = _atomic_path(predictions_path)
    try:
        with temporary_summary.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, default=_json_value)
            file.write("\n")

        with temporary_predictions.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=PREDICTION_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        os.replace(temporary_summary, summary_path)
        os.replace(temporary_predictions, predictions_path)
    finally:
        temporary_summary.unlink(missing_ok=True)
        temporary_predictions.unlink(missing_ok=True)

    return summary_path, predictions_path
