from typing import Callable, Dict, Any, Sequence

from preprocessing_functions import transpose_ffr

from pymatreader import read_mat

from EEGData import EEGData

from enum import Enum, auto

from user_functions import *

class ArgumentSpecification: 
    def __init__(self, parameter_label: str, parameter_name: str, data_type: int | float | str, default_value: Any, is_user_facing: bool):
        self.parameter_label = parameter_label
        self.parameter_name = parameter_name
        self.data_type = data_type
        self.default_value = default_value
        self.is_user_facing = is_user_facing # Specifies if the user specifies this argument

class FriendlyFunction:
    """
    Container for the GUI to interact with external functions.
    """
    def __init__(self, *, function: Callable, name: str, arguments: Sequence[ArgumentSpecification]):
        self.function = function
        self.name = name
        self.arguments = arguments
    
    def user_facing_arguments(self):
        return [arg for arg in self.arguments if arg.is_user_facing]

    def run(self, **kwargs):
        self.function(**kwargs)

if __name__ == "__main__":

    # Creating the subject 
    e = EEGSubject.init_from_filepath("/Users/kevin/Desktop/Work/SPAN_Lab/trial-classification/data/4T1002.mat")

    
    ff = FriendlyFunction(
        function=e.subaverage,
        name="subaverage",
        arguments=[
            ArgumentSpecification("Size", "size", int, 5, True)
        ]
    )

    d = {"size": 5}

    ff.run(**d)
    print(len(e._subaveraged_trials))