from sys import stdout

class StationaryPrinter:
    is_locked: bool = False
    
    @staticmethod
    def printl(printable):
        """
        Moves the cursor to the beginning of the line it is current on, prints the printable, 
        and does not create a new line afterward. 
        
        Use this print method consecutively to write text in-place instead of onto new lines.
        
        Parameters
        ----------
        printable
            The content to be printed. This can be anything compatible with the generic `print`.
        """
        StationaryPrinter.lock()
        print(f"\r{printable}", end="")
        stdout.flush()
    
    @staticmethod
    def print(printable):
        """
        Prints onto a new line like a normal `print` statement. 
        
        Use this method in replacement of `print` from builtins for compatability with `printl`. 
        
        Parameters
        ----------
        printable
            The content to be printed. This can be anything compatible with the generic `print`.
        """
        StationaryPrinter.unlock()
        print(printable)
    
    @classmethod
    def lock(cls) -> bool:
        """
        Locks the printer onto the current line and returns what the printer's locked status was 
        before the execution of this method. 
        
        Returns
        -------
        True if the printer was unlocked at the instant this function was called; false otherwise.
        """
        was_locked = cls.is_locked
        cls.is_locked = True
        return was_locked
    
    @classmethod
    def unlock(cls) -> bool:
        """
        Unlocks printer from the current line and returns what the printer's locked status was 
        before the execution of this method. 
        
        Creates a new line if the printer was previously locked, 
        since `printl` does not create a newline character. 
        
        Returns
        -------
        True if the printer was unlocked at the instant this function was called; false otherwise.
        """
        was_locked = cls.is_locked
        if was_locked:
            print()
        cls.is_locked = False
        return was_locked

if __name__ == "__main__":
    printl = StationaryPrinter.printl
    print = StationaryPrinter.print
    unlock = StationaryPrinter.unlock

    # Example usage
    print("Title")
    for i in range(1, 1000000):
        printl(f"First loop value: {i}")
    unlock() # Separates locked sections
    for i in range(1, 1000000):
        printl(f"Second loop value: {i}")
    print("Footer")