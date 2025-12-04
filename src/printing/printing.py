import sys

class StationaryPrinter:
    is_locked: bool = False
        
    @staticmethod
    def sprint(printable):
        if StationaryPrinter.is_locked:
            print(f"\r{printable}", end="")
            sys.stdout.flush()
        else:
            print(f"{printable}")
    
    @classmethod
    def lock(cls):
        cls.is_locked = True
    
    @classmethod
    def unlock(cls):
        cls.is_locked = False
        print()

if __name__ == "__main__":
    sprint = StationaryPrinter.sprint
    lock = StationaryPrinter.lock
    unlock = StationaryPrinter.unlock

    # Example usage
    sprint("Hello world!")
    lock()
    for i in range(1, 10000001):
        sprint(f"{i}")
    unlock()
    sprint("End")