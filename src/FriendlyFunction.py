from typing import Callable, Dict, Any

from preprocessing_functions import transpose_ffr

from pymatreader import read_mat

from EEGData import EEGData

from enum import Enum, auto

class FriendlyFunction:
    """
    Container for the GUI to interact with external functions.
    """

    class OperationType(Enum):
        DATA_LOADING = auto()
        DATA_TRANSFORMATION = auto()

    class SourceType(Enum):
        EEG_DATA = auto()
        ARGS = auto()
        EEG_DATA_AND_ARGS = auto()
        NA = auto()

    def __init__(self, *, function: Callable, name: str, usage: Dict[str, Dict[str, Any]], operation_type=OperationType.DATA_TRANSFORMATION, source_type=SourceType.NA):
        """
        :param function: The function to call.
        :param name: The name of the function.
        :param usage: The function is used. 
            The key is the parameter name, and the value 
            is the dictionary of the parameter type and default value.

            `Example: 
            {
                'num_folds': {
                    'type': 'int',
                    'default': 5
                },
                'num_trials': {
                    'type': 'int',
                    'default': 10
                }
            }
            `
        """
        self.function = function
        self.name = name
        self.usage = usage
        self.operation_type = operation_type
        self.source_type = source_type
    
    def run(self, on: EEGData=None, **kwargs):
        # Route based on source type: whether the function consumes EEGData context
        if self.source_type == FriendlyFunction.SourceType.EEG_DATA:
            # Prefer calling a bound method on the EEGData instance (self.data.method())
            if isinstance(self.function, str):
                method = getattr(on, self.function, None)
                if not callable(method):
                    raise AttributeError(f"EEGData has no callable attribute '{self.function}'")
                return method()
            # If a callable was provided, try to resolve a method of the same name on the instance
            if callable(self.function) and hasattr(self.function, '__name__'):
                candidate = getattr(on, self.function.__name__, None)
                if callable(candidate):
                    return candidate()
            # Fallback to passing the instance into the function
            return self.function(on)
        elif self.source_type == FriendlyFunction.SourceType.EEG_DATA_AND_ARGS:
            return self.function(on, **kwargs)
        else:
            return self.function(**kwargs)

def load_data_function(filename, into=None):
    raw_data = read_mat(filename)
    data = raw_data["ffr_nodss"]
    times = raw_data["time"]
    labels = raw_data["#subsystem#"]["MCOS"][3]
    
    # Create new EEGData instance
    new_data = EEGData(data=data, times=times, labels=labels)
    
    # If 'into' is provided and is a list/container, modify it in place
    if into is not None and hasattr(into, '__setitem__'):
        into[0] = new_data
    
    return new_data

f_load_data = FriendlyFunction(
    function=load_data_function,
    name="Load data",
    usage={
        "filename": {
            "type": str,
            "default": "/Users/kevin/Desktop/Work/SPAN_Lab/trial-classification/data/4T1002.mat"
        }
    },
    operation_type=FriendlyFunction.OperationType.DATA_LOADING,
    source_type=FriendlyFunction.SourceType.ARGS
)

f_transpose_ffr = FriendlyFunction(
    function="transpose_data",
    name="Transpose FFR",
    usage={},
    operation_type=FriendlyFunction.OperationType.DATA_TRANSFORMATION,
    source_type=FriendlyFunction.SourceType.EEG_DATA
)

f_test_function = FriendlyFunction(
    function=lambda name: print(f"Hello {name}!"),
    name="Print name",
    usage={
        "name": {
            "type": str,
            "default": "Charlie"
        }
    },
    source_type=FriendlyFunction.SourceType.ARGS
)