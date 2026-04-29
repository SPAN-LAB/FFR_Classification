"""
SPAN Lab - FFR Classification

Filename: printing.py
Author(s): Kevin Chen
Description: Defines functions that make it to make "stationary" prints.
    A stationary print is one where writing consecutively to the console overwrites its existing 
    content instead of printing onto a new line. 
"""


from sys import stdout

class Line:
    def __init__(self):
        self._has_placed = False
    
    def place(self, printable_content):
        # ANSI Codes:
        # \033[F - Move cursor to the beginning of the previous line
        # \033[K - Clear from cursor to end of line
        
        content_str = str(printable_content)
        
        if self._has_placed:
            # Move up 1 line, clear it, write new content, then go back to the next line
            # We add \n at the end so the cursor always sits on a fresh line for other prints
            stdout.write(f"\033[F\033[K{content_str}\n")
        else:
            # First time: just write and go to the next line
            stdout.write(f"{content_str}\n")
            self._has_placed = True
            
        stdout.flush()
    

class StationaryPrinter:
    is_locked: bool = False
    last_length: int = 0
    
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
        
        text = str(printable)
        StationaryPrinter.lock()

        # Clear previous text fully
        stdout.write("\r" + " " * StationaryPrinter.last_length)
        stdout.write("\r" + text)
        stdout.flush()

        StationaryPrinter.last_length = len(text)
    
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
        cls.last_length = 0
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