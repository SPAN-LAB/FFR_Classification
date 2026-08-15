"""Analysis APIs, loaded lazily to keep lightweight tools dependency-free."""

from importlib import import_module


__all__ = [
    "accuracy_against_subaverage_size",
    "accuracy_against_data_amount",
    "ConditionResult",
    "run_analysis_conditions",
]


def __getattr__(name):
    modules = {
        "accuracy_against_subaverage_size": ".accuracy_against_subaverage_size",
        "accuracy_against_data_amount": ".accuracy_against_data_amount",
        "ConditionResult": ".runner",
        "run_analysis_conditions": ".runner",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
