from typing import Callable

from ..core import AnalysisPipeline, PipelineState
from ..core.utils import FunctionKind

class Manager:
    def __init__(self):
        # Stores the source of truth for the GUI program
        self.state = PipelineState()
        self.initial_subjects_state = PipelineState()

        # List of tuples, each containing the name of the function and its parameters
        self.functions: list[tuple[str, dict]] = []

    def load_subjects(self, folder_path: str):
        # Reset the stored states
        self.state = PipelineState()
        self.initial_subjects_state = PipelineState()

        self.state.load_subjects(folder_path=folder_path)
        self.state.save(to=self.initial_subjects_state)

    def find_functions(self) -> dict[str, Callable]:
        """
        Returns a mapping from function names in AnalysisPipeline
        to the callable functions themselves. 

        IMPORTANT NOTE: Only functions whose kind is ``FunctionKind.gui`` are returned
            To see how functions get a ``kind`` attribute, see ``core/utils/details.py``
            and the function decorators in ``analysis_pipeline.py``.
        """
        mapping: dict[str, Callable] = {}
        for attr_name in dir(AnalysisPipeline):
            attr = getattr(AnalysisPipeline, attr_name)
            if not callable(attr):
                continue
            
            detail = getattr(attr, "detail", None)
            kind = getattr(detail, "kind", None)

            if kind is FunctionKind.gui:
                fn = getattr(self.state, attr_name)
                mapping[attr_name] = fn
        
        return mapping

    def run_function(self, name: str, **parameters):
        """
        Uses ``find_functions`` to find and then run a function.
        """
        func = self.find_functions()[name]
        return func(**parameters)
    
    def run_all_functions(self):
        for (function_name, parameters) in self.functions:
            self.run_function(function_name, **parameters)        
        

