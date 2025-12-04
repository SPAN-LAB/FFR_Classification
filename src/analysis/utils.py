import os

from ..core import AnalysisPipeline, PipelineState

def get_mats(folder_path: str) -> list[str]:
    """
    Returns a list of the paths to the `.mat` files contained in specified folder.
    """
    if not os.path.isdir(folder_path):
        raise ValueError("The path provided is not a folder path.")

    files = []
    for file in os.listdir(folder_path):
        if file.endswith(".mat"):
            files.append(os.path.join(folder_path, file))
    return files

def get_subject_loaded_pipelines(subject_filepaths: list[str]) -> dict[str, PipelineState]:
    """
    Returns a dictionary where each key is the filepath to a subject's data, and the value 
    associated with that key is an `AnalysisPipeline` object whose `subjects` attribute contains one
    `EEGSubject` which was loaded from the filepath. 
    
    Parameters
    ----------
    subject_filepaths : list[str]
        a list containing the filepaths to the subjects
    
    Returns
    -------
    dict[str, AnalysisPipeline]
        a dictionary where each value is an AnalysisPipeline loaded using its key
    """
    states = {}
    for subject_filepath in subject_filepaths:
        states[subject_filepath] = AnalysisPipeline().load_subjects(subject_filepath)
    return states