"""
SPAN Lab - FFR Classification

Filename: function_detail.py
Author(s): Kevin Chen
Description: The definition of FunctionDetail, the data type used to decorate functions to make them 
    GUI-friendly and GUI-representable.
"""


from __future__ import annotations
from enum import Enum, auto
from typing import Callable

class FunctionKind(Enum):
    gui = auto()
    gui_private = auto()
    non_gui = auto()

class Selection:
    def __init__(self, map: Callable[[], dict[str, any]]):
        self.option_value_map = map

    @property
    def options(self) -> list[str]:
        return list(self.option_value_map(self).keys())

    def value(self, selection: str):
        return self.option_value_map()[selection]

class FunctionDetail:
    def __init__(
        self,
        label: str,
        argument_details: list[ArgumentDetail],
        description: str | None = None,
        kind: FunctionKind = FunctionKind.gui
    ):
        self.label = label
        self.argument_details = argument_details
        self.description = description
        self.kind = kind

class ArgumentDetail:
    def __init__(
        self,
        label: str,
        type: type[int | float | str | dict[str, any] | Selection],
        default_value: int | float | str | dict[str, any] | Selection | None = None,
        description: str | None = None
    ):
        self.label = label
        self.type = type
        self.default_value = default_value
        self.description = description
