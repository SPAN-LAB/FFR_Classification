from typing import *

def function_label(label: str):
    """
    Set a GUI-ready function label.

    See below for usage
    """
    def decorator(function):
        function.label = label
        return function
    return decorator

def param_labels(labels: List[str]):
    """
    Conveniently set the GUI-ready parameter labels corresponding to a function's parameters.
    The order the labels must match the order of the function's parameters.

    See below for usage.
    """
    def decorator(function):
        function.parameter_labels = labels
        return function
    return decorator