from src.core import AnalysisPipeline
from src.core import EEGSubject
from src.core.plots import plot_averaged_trials

if __name__ == "__main__": 
    # Import subject data and setup
    from local.constants import GOOD_D_PATH
    subject15 = EEGSubject.init_from_filepath(filepath=GOOD_D_PATH)
    print(f"Subject initialized from file: {GOOD_D_PATH}")
    subject15.set_label_preference("raw")

    plot_averaged_trials(subject=subject15)