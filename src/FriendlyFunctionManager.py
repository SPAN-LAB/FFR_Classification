

from FriendlyFunction import FriendlyFunction, f_transpose_ffr, f_test_function, f_load_data
from typing import List

from EEGData import EEGData
from FriendlyFunction import FriendlyFunction as FF

class FriendlyFunctionManager:
    """
    A manager for GUI-callable functions.
    """
    def __init__(self, data: EEGData):
        self.data = data
        self.possible_functions: List[FriendlyFunction] = []
        self.functions: List[FriendlyFunction] = []

        # Add the external functions 
        self.register_function(f_transpose_ffr)
        self.register_function(f_test_function)
        self.register_function(f_load_data)
    
    def set_data(self, data: EEGData):
        self.data = data

    def add_function(self, function: FriendlyFunction):
        if type(function) != FriendlyFunction:
            raise ValueError("Function must be an instance of ExternalGUIFunction.")
        self.functions.append(function)

    def register_function(self, function: FriendlyFunction):
        if type(function) != FriendlyFunction:
            raise ValueError("Function must be an instance of ExternalGUIFunction.")
        self.possible_functions.append(function)
    
    def run_function(self, function_name: str, **kwargs):
        for function in self.possible_functions:
            if function.name == function_name:
                # Supply EEGData context only for functions that need it
                if getattr(function, 'source_type', None) == FF.SourceType.EEG_DATA:
                    result = function.run(on=self.data)
                elif getattr(function, 'source_type', None) == FF.SourceType.EEG_DATA_AND_ARGS:
                    result = function.run(on=self.data, **kwargs)
                else:
                    result = function.run(**kwargs)
                # If the function returned EEGData, update the manager's data regardless of operation type
                if isinstance(result, EEGData):
                    self.data = result
                return result
        raise ValueError(f"Function {function_name} not found in the manager.")

if __name__ == "__main__":
    # Get the EEGData 
    from pymatreader import read_mat
    filename = "/Users/kevin/Desktop/Work/SPAN_Lab/trial-classification/data/4T1002.mat"
    raw_data = read_mat(filename)
    data = raw_data["ffr_nodss"]
    times = raw_data["time"]
    labels = raw_data["#subsystem#"]["MCOS"][3]
    my_data = EEGData(data=data, times=times, labels=labels)

    # Create a new GUIFunctionManager
    fm = FriendlyFunctionManager(data=my_data)
    fm.run_function("Transpose FFR")
    # fm.run_function("Print provided name", name="Kevin")