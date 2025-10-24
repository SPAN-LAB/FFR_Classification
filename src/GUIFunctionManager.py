from __future__ import annotations
from typing import *
import inspect

import user_functions
import gui_functions
from Specifications import FunctionSpecification as FuncSpec
from Specifications import ArgumentSpecification as ArgSpec

class GUIFunctionManager: 
    def __init__(self):
        # Collect functions from user_functions (GLOBAL_*) and gui_functions (GUI_*)
        self.possible_functions: Dict[str, Callable] = {}
        self._label_to_source: Dict[str, str] = {}

        # User functions (eligible for pipeline builder)
        for function_name, function in inspect.getmembers(user_functions, inspect.isfunction):
            if function_name.startswith("GLOBAL"):
                self.possible_functions[function.label] = function
                self._label_to_source[function.label] = "user"

        # GUI utility functions (not shown in Add list; callable programmatically)
        for function_name, function in inspect.getmembers(gui_functions, inspect.isfunction):
            if function_name.startswith("GUI"):
                self.possible_functions[function.label] = function
                self._label_to_source[function.label] = "gui"

        # An ordered collection of the functions displayed in the GUI
        self.functions_arr: List[Callable] = []

    def get_possible_function_labels(self):
        # Only expose user-defined (pipeline) functions in the Add list
        return [label for label, src in self._label_to_source.items() if src == "user"]

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