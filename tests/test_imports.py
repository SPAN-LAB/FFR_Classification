"""Every package and model must import.

The model resolver skips modules that fail to import and only prints a warning,
so a broken model would otherwise disappear from the GUI and analyses silently.
"""

import importlib

import pytest

# `src.core` must be imported before the models; see the import order in
# `src/core/analysis_pipeline.py`.
import src.core  # noqa: F401
from src.models.utils import find_models, get_model_import_errors

# Models used by the analyses; each must stay discoverable by name.
ANALYSIS_MODELS = {
    "EEGNet",
    "EEGNeX",
    "EEGNeXFFR",
    "FFNN",
    "RNN",
    "LSTM",
    "GRU",
    "LDA",
    "SVM",
}


@pytest.mark.parametrize("package", ["src.core", "src.features", "src.analysis"])
def test_package_imports(package):
    importlib.import_module(package)


def test_analysis_api_resolves():
    analysis = importlib.import_module("src.analysis")
    for name in analysis.__all__:
        assert getattr(analysis, name) is not None


def test_every_model_module_imports():
    find_models()
    errors = get_model_import_errors()
    assert errors == {}, "Model modules failed to import:\n" + "\n".join(
        f"  {module}: {error}" for module, error in sorted(errors.items())
    )


def test_analysis_models_are_registered():
    missing = ANALYSIS_MODELS - set(find_models())
    assert not missing, f"Models missing from the registry: {sorted(missing)}"
