"""Loading subjects from MATLAB files."""

import numpy as np
import pytest
from scipy.io import savemat

from src.core import EEGSubject


def write_subject_file(path, n_trials=6, n_samples=20):
    data = np.arange(n_trials * n_samples, dtype=float).reshape(n_trials, n_samples)
    labels = np.array(["1", "2", "3"] * (n_trials // 3), dtype=object)
    savemat(path, {"ffr_nodss": data, "labels": labels, "time": np.linspace(0, 190, n_samples)})
    return data


def test_load_subject_from_mat_file(tmp_path):
    path = tmp_path / "S1.mat"
    data = write_subject_file(path)
    subject = EEGSubject.init_from_filepath(str(path))
    assert subject.name == "S1"
    assert len(subject.trials) == 6
    assert [trial.label for trial in subject.trials] == ["1", "2", "3", "1", "2", "3"]
    assert np.array_equal(subject.trials[1].data, data[1])
    assert len(subject.trials[0].timestamps) == 20


def test_missing_data_variable_raises_clear_error(tmp_path):
    path = tmp_path / "S1.mat"
    write_subject_file(path)
    with pytest.raises(ValueError, match="not found"):
        EEGSubject.init_from_filepath(str(path), data_var="not_a_variable")
