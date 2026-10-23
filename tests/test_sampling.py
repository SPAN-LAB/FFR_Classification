"""Balanced, seeded trial sampling used by the data-amount analyses."""

from collections import Counter

import pytest

from src.core.utils.sampling import sample_balanced_trials


def test_sample_is_balanced_with_no_duplicates(make_subject):
    trials = make_subject({"1": 20, "2": 20, "3": 20, "4": 20}).trials
    sampled = sample_balanced_trials(trials, 40, seed="test")
    assert Counter(trial.label for trial in sampled) == {"1": 10, "2": 10, "3": 10, "4": 10}
    assert len({trial.trial_index for trial in sampled}) == 40


def test_same_seed_gives_same_trials(make_subject):
    trials = make_subject({"1": 20, "2": 20}).trials
    first = sample_balanced_trials(trials, 10, seed="test")
    second = sample_balanced_trials(trials, 10, seed="test")
    assert [t.trial_index for t in first] == [t.trial_index for t in second]


def test_larger_sample_contains_smaller_sample(make_subject):
    trials = make_subject({"1": 20, "2": 20}).trials
    small = {t.trial_index for t in sample_balanced_trials(trials, 10, seed="test")}
    large = {t.trial_index for t in sample_balanced_trials(trials, 30, seed="test")}
    assert small <= large


def test_rejects_more_trials_than_a_label_has(make_subject):
    trials = make_subject({"1": 5, "2": 20}).trials
    with pytest.raises(ValueError):
        sample_balanced_trials(trials, 20, seed="test")
