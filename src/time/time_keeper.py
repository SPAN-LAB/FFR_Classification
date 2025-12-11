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
        