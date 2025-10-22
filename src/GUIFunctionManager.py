from __future__ import annotations
from typing import *
import inspect

import user_functions
from Specifications import FunctionSpecification as FuncSpec
from Specifications import ArgumentSpecification as ArgSpec

class GUIFunctionManager: 
    def __init__(self):
        # Get all the functions in user_functions.py
        # Except for the decorator.
        self.possible_functions: Dict[str, Callable] = {}
        for function_name, function in inspect.getmembers(user_functions, inspect.isfunction):
            if function_name.startswith("GLOBAL"):
                self.possible_functions[function.label] = function
        
        # An ordered collection of the functions displayd in the GUI
        self.functions_arr: List[Callable] = []

    def get_possible_function_labels(self):
        return [f.label for f in self.possible_functions.values()]

    def get_function_specification(self, function_label: str=None, function: Callable=None) -> FuncSpec:
        if not function_label and not function:
            raise ValueError("function_label and function cannot be None simultaneously.")
        elif function_label:
            function = self.possible_functions[function_label]
        return FuncSpec(function)

    def add_function(self, function_label: str=None, function: Callable=None):
        if not function_label and not function:
            raise ValueError("function_label and function cannot be None simultaneously.")
        elif function_label:
            function = self.possible_functions[function_label]

        self.functions_arr.append(function)

    def remove_function(self, function_label: str=None, function: Callable=None):
        if not function_label and not function:
            raise ValueError("function_label and function cannot be None simultaneously.")
        elif function_label:
            function = self.possible_functions[function_label]
        
        self.functions_arr.remove(function)

    def run_function(self, function_label: str, **kwargs):
        print("Running function: ", function_label)
        # Find the function 
        function = self.possible_functions[function_label]
        function(**kwargs)

    # Subject summaries have been moved to EEGSubject.summary(); GUI should query subjects directly

if __name__ == "__main__":
    f = GUIFunctionManager()
    f.run_function("Load Subject Data", filepath="/Users/kevin/Desktop/Work/SPAN_Lab/trial-classification/data/4T1002.mat")
    print(f.get_possible_function_labels())