from typing import Callable
import pickle
from PyQt5.QtCore import QObject, pyqtSignal

from ..core import AnalysisPipeline, PipelineState
from ..core.utils import FunctionKind

class Manager(QObject):
    # Signals for various events
    subjects_loaded = pyqtSignal(str)  # emits: folder_path
    pipeline_loaded = pyqtSignal(str)  # emits: file_path
    pipeline_created = pyqtSignal(str)  # emits: file_path
    pipeline_saved = pyqtSignal()
    function_added = pyqtSignal(str, dict)  # emits: name, parameters
    functions_updated = pyqtSignal()  # emits when functions list changes
    state_changed = pyqtSignal()
    
    def __init__(self):
        super().__init__()  # Initialize QObject
        
        # Stores the source of truth for the GUI program
        self.state = PipelineState()
        self.initial_subjects_state = PipelineState()

        # List of tuples, each containing the name of the function and its parameters
        self.functions: list[tuple[str, dict]] = []

        self.subjects_folder_path: str | None = None
        self.pipeline_path: str | None = None

    def load_subjects(self, folder_path: str):
        # Reset the stored states
        self.state = PipelineState()
        self.initial_subjects_state = PipelineState()

        self.state.load_subjects(folder_path)
        self.state.save(to=self.initial_subjects_state)

        # Update state and emit signals
        self.subjects_folder_path = folder_path
        self.subjects_loaded.emit(folder_path)
        self.state_changed.emit()

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
                mapping[detail.label] = fn
        
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
        

    def load_pipeline(self, filepath: str):
        """
        Load a pipeline from a pickle file.
        
        :param filepath: Path to the .pkl file
        :raises ValueError: If the file doesn't have .pkl extension
        """
        # Check file extension
        if not filepath.endswith('.pkl'):
            raise ValueError(f"Pipeline file must have .pkl extension, got: {filepath}")
        
        # Load the pickle file
        with open(filepath, 'rb') as f:
            self.functions = pickle.load(f)
        
        # Update state and emit signals
        self.pipeline_path = filepath
        self.pipeline_loaded.emit(filepath)
        self.functions_updated.emit()
        self.state_changed.emit()

    def create_pipeline(self, filepath: str):
        """
        Create a new pipeline by clearing functions and setting the pipeline path.
        
        :param filepath: Path where the pipeline will be saved
        """
        # Clear functions
        self.functions = []
        
        # Set pipeline path
        self.pipeline_path = filepath

        self.save_pipeline()
        
        # Emit signals
        self.pipeline_created.emit(filepath)
        self.functions_updated.emit()
        self.state_changed.emit()
    
    def save_pipeline(self):
        """
        Save the current pipeline to the pickle file at self.pipeline_path.
        
        :raises ValueError: If pipeline_path is not set
        """
        if self.pipeline_path is None:
            raise ValueError("No pipeline path set. Create or load a pipeline first.")
        
        # Save to pickle file
        with open(self.pipeline_path, 'wb') as f:
            pickle.dump(self.functions, f)
        
        # Emit signal
        self.pipeline_saved.emit()
    
    def add_function(self, name: str, parameters: dict):
        """
        Add a function to the pipeline.
        
        :param name: Function name
        :param parameters: Function parameters as a dictionary
        """
        self.functions.append((name, parameters))
        self.function_added.emit(name, parameters)
        self.functions_updated.emit()
        self.state_changed.emit()
    
    def remove_function(self, index: int):
        """
        Remove a function from the pipeline by index.
        
        :param index: Index of the function to remove
        """
        if 0 <= index < len(self.functions):
            self.functions.pop(index)
            self.functions_updated.emit()
            self.state_changed.emit()
    
    def clear_functions(self):
        """Clear all functions from the pipeline."""
        self.functions = []
        self.functions_updated.emit()
        self.state_changed.emit()
    
    def reorder_functions(self, old_index: int, new_index: int):
        """
        Reorder functions in the pipeline.
        
        :param old_index: Current index of the function
        :param new_index: New index for the function
        """
        if 0 <= old_index < len(self.functions) and 0 <= new_index < len(self.functions):
            item = self.functions.pop(old_index)
            self.functions.insert(new_index, item)
            self.functions_updated.emit()
            self.state_changed.emit()