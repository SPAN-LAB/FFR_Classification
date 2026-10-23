"""A short training run on synthetic data for one scikit-learn and one PyTorch model."""

import numpy as np
import pytest
import torch

from src.models.utils import find_model

TRAINING_OPTIONS = {"num_epochs": 2, "batch_size": 8, "learning_rate": 1e-3, "weight_decay": 0.0}


def noisy_subject(make_subject, name, seed):
    subject = make_subject({"1": 20, "2": 20}, n_samples=32, name=name)
    rng = np.random.default_rng(seed)
    for trial in subject.trials:
        trial.data = rng.normal(size=32) + (1.0 if trial.label == "2" else 0.0)
    return subject


@pytest.mark.slow
@pytest.mark.parametrize("model_name", ["LDA", "FFNN"])
def test_model_trains_and_predicts_every_trial(model_name, make_subject):
    torch.manual_seed(0)
    train = noisy_subject(make_subject, "train", seed=0)
    test = noisy_subject(make_subject, "test", seed=1)

    model = find_model(model_name)(dict(TRAINING_OPTIONS))
    if hasattr(model, "set_device"):
        model.set_device(use_gpu=False)
    model.set_subject(train)
    model.train(trials=list(train.trials))
    model.infer(trials=test.trials)

    for trial in test.trials:
        assert trial.prediction in {"1", "2"}
        assert sum(trial.prediction_distribution.values()) == pytest.approx(1.0, abs=1e-5)
