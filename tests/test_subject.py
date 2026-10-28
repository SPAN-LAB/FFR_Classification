"""Subaveraging, folding, labels, trimming, and merging on EEGSubject."""

import numpy as np
import pytest

from src.core import EEGSubject


def test_subaverage_averages_consecutive_trials_within_each_label(make_subject):
    subject = make_subject({"1": 6, "2": 4})
    subject.subaverage(2)
    by_label = subject.grouped_trials()
    # Label "1" holds trials 0-5, label "2" trials 6-9; each value is a pair's mean.
    assert [trial.data[0] for trial in by_label["1"]] == [0.5, 2.5, 4.5]
    assert [trial.data[0] for trial in by_label["2"]] == [6.5, 8.5]


def test_subaverage_drops_incomplete_groups(make_subject):
    subject = make_subject({"1": 5})
    subject.subaverage(2)
    assert len(subject.trials) == 2


def test_subaverage_size_one_keeps_every_trial(make_subject):
    subject = make_subject({"1": 3, "2": 3})
    original = [trial.data for trial in subject.trials]
    subject.subaverage(1)
    assert len(subject.trials) == 6
    assert all(np.array_equal(a, b.data) for a, b in zip(original, subject.trials))


def test_subaverage_rejects_size_zero(make_subject):
    with pytest.raises(ValueError):
        make_subject({"1": 3}).subaverage(0)


def test_fold_puts_each_trial_in_exactly_one_fold(make_subject):
    subject = make_subject({"1": 10, "2": 7})
    subject.fold(5)
    indices = [trial.trial_index for fold in subject.folds for trial in fold]
    assert sorted(indices) == list(range(17))


def test_every_fold_has_every_label(make_subject):
    subject = make_subject({"1": 10, "2": 7})
    subject.fold(5)
    for fold in subject.folds:
        assert {trial.label for trial in fold} == {"1", "2"}


def test_fold_rejects_too_few_trials(make_subject):
    with pytest.raises(ValueError):
        make_subject({"1": 3}).fold(5)


def test_labels_map_numbers_labels_in_order(make_subject):
    subject = make_subject({"1": 2, "2": 2, "3": 2})
    assert subject.labels_map == {"1": 0, "2": 1, "3": 2}
    assert subject.trials[-1].enumerated_label == 2


def test_map_trial_labels_applies_rules(make_subject, tmp_path):
    subject = make_subject({1: 2, 2: 2, 3: 2})
    rules = tmp_path / "rules.csv"
    rules.write_text("1, high\n2, high\n3, low\n")
    subject.map_trial_labels(str(rules))
    assert [trial.label for trial in subject.trials] == ["high"] * 4 + ["low"] * 2
    assert [trial.raw_label for trial in subject.trials] == [1, 1, 2, 2, 3, 3]
    assert set(subject.labels_map) == {"high", "low"}


def test_map_trial_labels_rejects_unmapped_label(make_subject, tmp_path):
    subject = make_subject({1: 2, 2: 2})
    rules = tmp_path / "rules.csv"
    rules.write_text("1, high\n")
    with pytest.raises(ValueError):
        subject.map_trial_labels(str(rules))


def test_trim_by_timestamp_keeps_matching_samples(make_subject):
    subject = make_subject({"1": 1}, n_samples=10)
    trial = subject.trials[0]
    trial.data = np.arange(10, dtype=float)
    trial.timestamps = np.arange(10, dtype=float) * 10  # 0, 10, ..., 90 ms
    subject.trim_by_timestamp(20, 50)
    assert list(trial.timestamps) == [20, 30, 40, 50]
    assert list(trial.data) == [2, 3, 4, 5]


def test_create_merged_keeps_every_trial_and_leaves_originals_alone(make_subject):
    first = make_subject({"1": 2, "2": 2}, name="A")
    second = make_subject({"1": 3, "2": 1}, name="B")
    merged = EEGSubject.create_merged(subjects=[first, second])
    assert len(merged.trials) == 8
    assert sorted(trial.label for trial in merged.trials) == ["1"] * 5 + ["2"] * 3
    assert all(trial.subject is merged for trial in merged.trials)
    assert all(trial.subject is first for trial in first.trials)
    assert all(trial.subject is second for trial in second.trials)
