"""
SPAN Lab - FFR Classification

Filename: details.py
Author(s): Kevin Chen
Description: Decorators for the methods of AnalysisPipeline that provide additional information 
    on a function for a GUI to represent it accurately. 
"""

"""THIS IS UNMAINTAINED"""

from .function_detail import FunctionDetail as FD
from .function_detail import ArgumentDetail as AD
from .function_detail import FunctionKind
from .function_detail import Selection

# from ...models.utils import find_models

def detail(detail: FD):
    def decorator(func):
        func.detail = detail
        return func
    return decorator

def undetailed():
    def decorator(func):
        func.detail = FD(label="",argument_details=[], kind=FunctionKind.non_gui)
        return func
    return decorator

def gui_private():
    def decorator(func):
        func.detail = FD(label="",argument_details=[], kind=FunctionKind.gui_private)
        return func
    return decorator

map_labels_detail = FD(
    label="Map Labels",
    argument_details=[
        AD(
            label="CSV Filepath",
            type=str,
            default_value="",
            description="The path to a CSV file specifying how the labels are mapped."
        )
    ],
    description="Maps the labels of each trial of each subject according to the provided file."
)

trim_by_timestamp_detail = FD(
    label="Trim by Timestamp",
    argument_details=[
        AD(
            label="Start Time (ms)",
            type=float,
            default_value=0,
            description="The lower bound for the timestamps."
        ),
        AD(
            label="End Time (ms)",
            type=float,
            default_value=100,
            description="The upper bound for the timestamps."
        )
    ],
    description="Keeps only the datapoints recorded between the provided timestamps."
)

trim_by_index_detail = FD(
    label="Trim by Index",
    argument_details=[
        AD(
            label="Start Index",
            type=int,
            default_value=0,
            description="The starting index (inclusive) of the trimmed sequence of datapoints."
        ),
        AD(
            label="End Index",
            type=int,
            default_value=None, # Explicitly set no default value
            description="The ending index (inclusive) of the trimmed sequence of datapoints."
        )
    ],
    description="Keeps only the datapoints recorded between the provided indices."
)

trim_by_type_detail = FD(
    label="Trim by Type",
    argument_details=[
        AD(
            label="Labels to Keep",
            type=str,
            default_value="1,2,3",
            description="Comma-separated labels to keep."
        ),
        AD(
            label="Label Source",
            type=str,
            default_value="raw",
            description="Use raw, mapped, or current labels."
        )
    ],
    description="Keeps only selected trial categories before classification."
)

subaverage_detail = FD(
    label="Subaverage Trials",
    argument_details=[
        AD(
            label="Number of Trials",
            type=int,
            default_value=5,
            description="The number of trials to combine through subaveraging."
        )
    ],
    description="Combines trials through subaveraging. This can help reduce noise in your data."
)

fold_detail = FD(
    label="Split into Folds",
    argument_details=[
        AD(
            label="Number of Folds",
            type=int,
            default_value=5,
            description="The number of groups to split the subject's trials into."
        )
    ],
    description="Divides each subject's trials into the number of groups (folds) provided."
)

extract_features_detail = FD(
    label="Extract Features",
    argument_details=[
        AD(
            label="Feature Names (comma-separated)",
            type=str,
            default_value="pitchtrack,autocorr,zerocrossing",
            description="Comma-separated features to compute: pitchtrack, autocorr, zerocrossing."
        )
    ],
    description="Computes one or more selected features for every loaded trial."
)

save_state_detail = FD(
    label="Save Full State",
    argument_details=[
        AD(
            label="Output Filepath",
            type=str,
            default_value="ffr_pipeline_state.pkl",
            description="Pickle filepath for the full pipeline state."
        )
    ],
    description="Saves loaded subjects, features, predictions, and models."
)

load_state_detail = FD(
    label="Load Full State",
    argument_details=[
        AD(
            label="Input Filepath",
            type=str,
            default_value="ffr_pipeline_state.pkl",
            description="Pickle filepath written by Save Full State."
        )
    ],
    description="Restores a full saved pipeline state."
)

save_features_detail = FD(
    label="Save Features",
    argument_details=[
        AD(
            label="Output Filepath",
            type=str,
            default_value="ffr_features.pkl",
            description="Pickle filepath for extracted features and feature inputs."
        )
    ],
    description="Saves extracted feature arrays and the raw inputs used to compute them."
)

load_features_detail = FD(
    label="Load Features",
    argument_details=[
        AD(
            label="Input Filepath",
            type=str,
            default_value="ffr_features.pkl",
            description="Pickle filepath written by Save Features."
        )
    ],
    description="Loads saved feature arrays onto matching loaded subjects."
)

save_visualization_data_detail = FD(
    label="Save Visualization Data",
    argument_details=[
        AD(
            label="Output Filepath",
            type=str,
            default_value="ffr_visualization_data.pkl",
            description="Pickle filepath for data needed to visualize later."
        )
    ],
    description="Saves trial data, labels, features, predictions, and probabilities."
)

load_visualization_data_detail = FD(
    label="Load Visualization Data",
    argument_details=[
        AD(
            label="Input Filepath",
            type=str,
            default_value="ffr_visualization_data.pkl",
            description="Pickle filepath written by Save Visualization Data."
        )
    ],
    description="Loads saved subjects and prediction data for plotting."
)

evaluate_model_detail = FD(
    label="Evaluate Model",
    argument_details=[
        # AD(
        #     label="Select Model",
        #     type=Selection,
        #     default_value=Selection(map=find_models),
        #     description="Select your model here."
        # ),
        
        AD(
            label="Training Options",
            type=dict[str, any],
            default_value={
                "num_epochs": 20,
                "batch_size": 32,
                "learning_rate": 0.001,
                "weight_decay": 0.1
            }
        )
    ]
)

# TODO
train_model_detail = FD(
    label="Train Model",
    argument_details=[],
    description="This function is NOT YET implemented."
)

# TODO
infer_on_model_detail = FD(
    label="Infer on Model",
    argument_details=[],
    description="This function is NOT YET implemented."
)
