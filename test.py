from src.core import EEGSubject

subject1: EEGSubject = EEGSubject.init_from_filepath("data/4T1002.mat")
subject1.trim_by_timestamp(start_time=5, end_time=6)
print(subject1.trials[0].timestamps)