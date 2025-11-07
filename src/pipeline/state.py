from src.core import EEGSubject
import os

class PilepineState: 
    def __init__(self):
        self.subjects = []

    def load_subjects(self, folder_path: str):
        """
        Loads all subjects from the input folder path.
        """

        for file in os.listdir(folder_path):
            if file.endswith(".mat"):
                subject = EEGSubject.init_from_filepath(os.path.join(folder_path, file))
                self.subjects.append(subject)
        
    def subaverage(self, size: int):
        """
        Subaverages all subjects in the pipeline.
        """
        for subject in self.subjects:
            subject.subaverage(size)
    
