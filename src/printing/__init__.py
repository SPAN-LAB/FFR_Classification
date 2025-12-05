from .printing import StationaryPrinter

printl = StationaryPrinter.printl
print = StationaryPrinter.print
lock = StationaryPrinter.lock
unlock = StationaryPrinter.unlock

__all__ = [
    "printl"
    "print",
    "lock",
    "unlock"
]