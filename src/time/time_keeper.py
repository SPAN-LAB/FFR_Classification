"""
SPAN Lab - FFR Classification

Filename: time_keeper.py
Author(s): Kevin Chen
Description: A simple timer; useful for measuring how long some computation takes to complete. 
    See example code below.
"""


import time

class TimeKeeper:
    def __init__(self):
        self.start = time.perf_counter()
        self.last = time.perf_counter()
    
    def lap_time(self) -> float:
        """
        Record a lap and get the time since the last lap.
        """
        curr_time = time.perf_counter() 
        duration = curr_time - self.last
        self.last = curr_time
        return duration
    
    def end_time(self) -> float:
        """
        Get the time since the start of time-keeping.
        """
        return time.perf_counter() - self.start
    
    def peek_time(self) -> float:
        return self.end_time()

if __name__ == "__main__": 
    
    tk = TimeKeeper()
    
    for _ in range(1_000_000):
        # ... some computation ...
        print(f"Time since previous iteration: {tk.lap_time()} s")
    
    print(f"Total duration: {tk.end_time()}")
    