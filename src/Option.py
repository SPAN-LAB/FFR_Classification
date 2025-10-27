from __future__ import annotations
from typing import Any, Callable
from protocol import EEGSubjectProtocol

class Option:
    """
    Represents the possible options that can be made.
    """
    options: list[Any]

    def __init__(self, *args: Any):
        self.options = list(args)

class ComputableOption(Option):
    options: Callable[[list[EEGSubjectProtocol]], list[Any]]

    def __init__(self, options: Callable[[list[EEGSubjectProtocol]], list[Any]], subjects: list[EEGSubjectProtocol]):
        self.options = options
        self.subjects = subjects