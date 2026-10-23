"""Shared helpers for building small synthetic subjects."""

import numpy as np
import pytest

# `src.core` must be imported before the models; see the import order in
# `src/core/analysis_pipeline.py`.
import src.core  # noqa: F401
from src.core import EEGSubject, EEGTrial


def _make_subject(counts, n_samples=8, name="S1"):
    """Build a subject with `counts[label]` trials per label.

    Trial i's waveform is all i, so averages are easy to check by hand.
    """
    subject = EEGSubject(source_filepath=f"{name}.mat")
    for label, count in counts.items():
        for _ in range(count):
            index = len(subject.trials)
            subject.trials.append(
                EEGTrial(
                    subject=subject,
                    data=np.full(n_samples, float(index)),
                    trial_index=index,
                    timestamps=np.arange(n_samples, dtype=float),
                    raw_label=label,
                )
            )
    subject.setup_labels_map()
    return subject


@pytest.fixture
def make_subject():
    return _make_subject
