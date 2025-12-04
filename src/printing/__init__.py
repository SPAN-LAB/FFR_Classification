from .printing import StationaryPrinter

sprint = StationaryPrinter.sprint
lprint = StationaryPrinter.lprint
ulprint = StationaryPrinter.ulprint
lock = StationaryPrinter.lock
unlock = StationaryPrinter.unlock

__all__ = [
    "StationaryPrinter",
    "sprint",
    "lprint",
    "ulprint",
    "lock",
    "unlock"
]