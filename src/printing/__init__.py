from .printing import StationaryPrinter

sprint = StationaryPrinter.sprint
lock = StationaryPrinter.lock
unlock = StationaryPrinter.unlock

__all__ = [
    "StationaryPrinter",
    "sprint",
    "lock",
    "unlock"
]