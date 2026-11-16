"""Per-subject data-amount analysis at different subaverage sizes."""

import json
import sys

import numpy as np
import pytest
from scipy.io import savemat

from server_analysis import run_analysis
from src.analysis.runner import run_analysis_conditions

TRIALS_PER_TONE = 50


@pytest.fixture
def subject_file(tmp_path):
    rng = np.random.default_rng(0)
    labels = np.array(["1", "2", "3", "4"] * TRIALS_PER_TONE, dtype=object)
    data = rng.normal(size=(len(labels), 64))
    path = tmp_path / "S1.mat"
    savemat(path, {"ffr_nodss": data, "labels": labels, "time": np.linspace(0, 300, 64)})
    return path


def count_test_examples(subject_file, size):
    [result] = run_analysis_conditions(
        model_name="LDA",
        subject_filepath=str(subject_file),
        analysis="data_amount",
        values=[80],  # LDA needs more examples than tones, even at size 10
        data_amount_subaverage_size=size,
    )
    assert result.status == "success", result.error_traceback
    return len(result.pipeline.subjects[0].trials)


def test_subaverage_size_sets_how_many_test_examples_there_are(subject_file):
    # Each test fold holds 10 raw trials per tone (50 / 5 folds).
    assert count_test_examples(subject_file, 1) == 4 * TRIALS_PER_TONE
    assert count_test_examples(subject_file, 5) == 4 * 5 * 2
    assert count_test_examples(subject_file, 10) == 4 * 5 * 1


def test_cli_writes_explicit_sizes_to_their_own_folder(subject_file, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "run_analysis", "--model", "LDA", "--subject", str(subject_file),
        "--analysis", "data_amount", "--value", "80",
        "--data-amount-subaverage-size", "10", "--output-dir", str(tmp_path / "out"),
    ])
    run_analysis.main()
    summary_path = (
        tmp_path / "out" / "data_amount_by_subaverage" / "subaverage_10" / "LDA"
        / "LDA.S1.data_amount.sa10-80.summary.json"
    )
    summary = json.loads(summary_path.read_text())
    assert summary["data_amount_subaverage_size"] == 10
    assert summary["status"] == "success"


def test_cli_rejects_size_for_subaverage_analysis(subject_file, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "run_analysis", "--model", "LDA", "--subject", str(subject_file),
        "--analysis", "subaverage", "--data-amount-subaverage-size", "10",
    ])
    with pytest.raises(ValueError, match="only valid with --analysis data_amount"):
        run_analysis.main()
