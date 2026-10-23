"""Turning model outputs back into labels, and computing accuracy."""

import pytest

from src.core import EEGTrial


def test_prediction_index_maps_back_to_label(make_subject):
    subject = make_subject({"1": 1, "2": 1, "3": 1})
    trial = subject.trials[0]
    trial.set_prediction_distribution(enumerated_prediction_distribution={0: 0.1, 1: 0.7, 2: 0.2})
    assert trial.prediction == "2"
    assert trial.prediction_distribution == {"1": 0.1, "2": 0.7, "3": 0.2}


def test_accuracy_counts_correct_predictions(make_subject):
    trials = make_subject({"1": 2, "2": 2}).trials
    for trial, prediction in zip(trials, ["1", "2", "2", "2"]):
        trial.prediction = prediction
    assert EEGTrial.get_accuracy(trials) == pytest.approx(0.75)
