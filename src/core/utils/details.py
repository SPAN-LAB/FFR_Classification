from .function_detail import FunctionDetail as FD
from .function_detail import ArgumentDetail as AD
from .function_detail import Selection

def detail(detail: FD):
    def decorator(func):
        func.detail = detail
        return func
    return decorator

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
