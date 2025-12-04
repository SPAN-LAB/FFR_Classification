from sys import stdout

class StationaryPrinter:
    is_locked: bool = False
 
    @staticmethod
    def sprint(printable):
        if StationaryPrinter.is_locked:
            print(f"\r{printable}", end="")
            stdout.flush()
        else:
            print(f"{printable}")
    
    @staticmethod
    def lprint(printable):
        if not StationaryPrinter.lock():
            print()
        StationaryPrinter.sprint(printable)
    
    @staticmethod
    def ulprint(printable):
        if StationaryPrinter.unlock(): # if changed from locked
            print()
        StationaryPrinter.sprint(printable)
    
    @classmethod
    def lock(cls) -> bool:
        prev = cls.is_locked
        cls.is_locked = True
        return prev
    
    @classmethod
    def unlock(cls) -> bool:
        prev = cls.is_locked
        cls.is_locked = False
        return prev

if __name__ == "__main__":
    ulprint = StationaryPrinter.ulprint
    lprint = StationaryPrinter.lprint

    # Example usage
    ulprint("Hello")
    for i in range(1, 100001):
        lprint(f"{i}")
    ulprint("End")