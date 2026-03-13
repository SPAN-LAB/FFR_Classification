"""
SPAN Lab - FFR Classification

Filename: time_keeper.py
Author(s): Kevin Chen
Description: A simple timer; useful for measuring how long some computation takes to complete. 
    See example code below.
"""


import time

class TimeKeeper:
    def __init__(self, num_dps: int):
        self.start_time = 0
        self.last_resume_time = 0
        self.last_lap_duration = 0
        self.accumulated_duration = 0
        self.reset()
        self.num_dps = num_dps

    def reset(self):
        self.start_time = time.perf_counter()
        self.last_resume_time = time.perf_counter()
        self.last_lap_duration = 0
        self.accumulated_duration = 0

    def start(self):
        self.start_time = time.perf_counter()
        
    def lap(self):
        temp_last_lap_duration = time.perf_counter() - self.last_resume_time
        if self.num_dps:
            self.last_lap_duration = round(temp_last_lap_duration, self.num_dps)
        else:
            self.last_lap_duration = temp_last_lap_duration
        
        self.last_resume_time = time.perf_counter()
    
    def stop(self):
        temp_accumulated_duration = time.perf_counter() - self.start_time
        if self.num_dps:
            self.accumulated_duration = round(temp_accumulated_duration, self.num_dps)
        else:
            self.accumulated_duration = temp_accumulated_duration

if __name__ == "__main__": 
    
    tk = TimeKeeper(num_dps=2)
    
    tk.reset()
    tk.start()
    time.sleep(10)
    tk.stop()
    print(f"{tk.accumulated_duration}")