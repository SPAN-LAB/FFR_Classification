"""
SPAN Lab - FFR Classification

Filename: demo.py
Author(s): Kevin Chen
Description: Example code for using the AnalysisPipeline APIs.
"""


from src.core import AnalysisPipeline

# UPDATE ME
DIR_OR_FILE_PATH = None

def extract(dictionary):

    # Data is a 2D array with rows corresponding to trials 
    # and columns corresponding to samples
    
    return {
        "data": dictionary["a"],
        "timestamps": dictionary["b"],
        "labels": dictionary["c"]
    }
    
p = (
    AnalysisPipeline()
    .load_subjects(DIR_OR_FILE_PATH, extract=extract)
    .trim_by_timestamp(start_time=50, end_time=250) # Keep all starting from 0 ms
    .subaverage(5)
    .fold(5)
    .evaluate_model(
        model_name="CNN",
        training_options={
            "num_epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.001,
            "weight_decay": 0.1
        }
    )
)
