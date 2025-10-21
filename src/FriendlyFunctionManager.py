from __future__ import annotations
from typing import *

from EEGDataStructures import EEGSubject
from FriendlyFunction import FriendlyFunction 
from FriendlyFunction import ArgumentSpecification as ArgSpec

class FriendlyFunctionManager:
    """
    A manager for GUI-callable functions.
    """
    def __init__(self, e: EEGSubject=None):
        self.e = e
        self.possible_functions: Sequence[FriendlyFunction] = []
        self.functions: Sequence[FriendlyFunction] = []

        def compound_load(filepath: str):
            _e = EEGSubject.init_from_filepath(filepath)
            if len(self.possible_functions) == 1:
                self.possible_functions.append(FriendlyFunction(
                    function=_e.subaverage,
                    name="Subaverage Trials",
                    arguments=[
                        ArgSpec(
                            parameter_label="Size",
                            parameter_name="size",
                            data_type=int,
                            default_value=5,
                            is_user_facing=True,
                        )
                    ]
                ))
                self.possible_functions.append(FriendlyFunction(
                    function=_e.stratified_folds,
                    name="Split Trials into Stratified Folds",
                    arguments=[
                        ArgSpec(
                            parameter_label="Number of Folds",
                            parameter_name="num_folds",
                            data_type=int,
                            default_value=5,
                            is_user_facing=True,
                        )
                    ]
                ))
            if self.e is None:
                self.e = _e

        load_data_function = FriendlyFunction(
            function=compound_load,
            name="Load Data from File",
            arguments=[
                ArgSpec(
                    parameter_label="Filepath",
                    parameter_name="filepath",
                    data_type=str,
                    default_value="/Users/kevin/Desktop/Work/SPAN_Lab/trial-classification/data/4T1002.mat",
                    is_user_facing=True,
                )
            ],
        )
        self.possible_functions.append(load_data_function)

    @property
    def available_functions(self):
        # If nothing registered yet
        if len(self.possible_functions) == 0:
            return []
        # Loader is first; if no subject loaded, only expose loader
        if self.e is None:
            return [self.possible_functions[0]]
        # Subject present: expose everything except the loader
        return self.possible_functions[1:]

    def run_function(self, function_name: str, **kwargs):
        function = None
        for f in self.possible_functions:
            if f.name == function_name:
                function = f
                break
        
        if function is None:
            raise ValueError(f"Function {function_name} not found")
        function.run(**kwargs)

if __name__ == "__main__":
    f = FriendlyFunctionManager()
    f.run_function("load_data", filepath="/Users/kevin/Desktop/Work/SPAN_Lab/trial-classification/data/4T1002.mat")
    f.run_function("subaverage", size=5)
    print(len(f.e.trials))