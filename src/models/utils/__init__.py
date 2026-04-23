from .model_interface import ModelInterface
from .resolver import find_model, find_models

# Lazy import for torch-dependent classes to avoid import errors when torch is not available
def __getattr__(name):
    if name == "TorchNNBase":
        from .torchnn_base import TorchNNBase
        return TorchNNBase
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")