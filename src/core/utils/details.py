from .function_detail import FunctionDetail as FD
from .function_detail import ArgumentDetail as AD
from .function_detail import FunctionKind
from .function_detail import Selection

from ...models.utils import find_models

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

evaluate_model_detail = FD(
    label="Evaluate Model",
    argument_details=[
        AD(
            label="Select Model",
            type=Selection,
            default_value=Selection(map=find_models),
            description="Select your model here."
        ),
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
