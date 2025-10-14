from typing import Callable, Dict, Any, List

class ExternalGUIFunction:
    """
    Container for externally-registered GUI functions.
    """

    def __init__(self, *, function_name: str, callback: Callable, function_args: Dict[str, Any], guidance: Dict[str, type] | None = None):
        self.function_name = function_name
        self.callback = callback
        self.function_args = function_args
        # Guidance is a mapping of parameter name -> type (int, float, str)
        self.guidance = guidance or {}

class GUIFunctionManager:
    """
    Manages GUI-callable functions. Supports built-in class-level functions and externally
    added functions for extensibility.
    """

    # Parameter guidance per builtin function
    builtin_function_guidance: Dict[str, Dict[str, type]] = {
        "print_stuff": {"number": float, "name": str},
    }

    # Registry for externally-added functions and their guidance
    external_functions: Dict[str, Callable] = {}
    external_function_guidance: Dict[str, Dict[str, type]] = {}

    @staticmethod
    def print_stuff(*, number: int | float | str, name: str) -> None:
        print(f"Number: {number}, Name: {name}")

    def __init__(self):
        self.functions: List[ExternalGUIFunction] = []
    
    def add_gui_function(self, external_gui_function: ExternalGUIFunction) -> None:
        self.functions.append(external_gui_function)
    
    def run_functions(self) -> None:
        for function in self.functions:
            function.callback(**function.function_args)
    
    def get_function_args(self, function_name: str) -> Dict[str, Any] | None:
        for function in self.functions:
            if function.function_name == function_name:
                return function.function_args
        return None

    @classmethod
    def get_builtin_callable(cls, function_name: str) -> Callable | None:
        # Backward-compat: delegate to unified resolver
        return cls.get_callable(function_name)

    @classmethod
    def get_builtin_guidance(cls, function_name: str) -> Dict[str, type] | None:
        # Backward-compat: delegate to unified guidance lookup
        return cls.get_guidance(function_name)

    @classmethod
    def register_function(cls, *, function_name: str, callback: Callable, guidance: Dict[str, type] | None = None) -> None:
        cls.external_functions[function_name] = callback
        if guidance is not None:
            cls.external_function_guidance[function_name] = guidance
        else:
            cls.external_function_guidance.setdefault(function_name, {})

    @classmethod
    def unregister_function(cls, function_name: str) -> None:
        cls.external_functions.pop(function_name, None)
        cls.external_function_guidance.pop(function_name, None)

    @classmethod
    def get_callable(cls, function_name: str) -> Callable | None:
        # Built-in
        if function_name in cls.builtin_function_guidance:
            candidate = getattr(cls, function_name, None)
            return candidate if callable(candidate) else None
        # External
        return cls.external_functions.get(function_name)

    @classmethod
    def get_guidance(cls, function_name: str) -> Dict[str, type] | None:
        return cls.builtin_function_guidance.get(function_name) or cls.external_function_guidance.get(function_name)

    @classmethod
    def list_available_function_names(cls) -> List[str]:
        return sorted(set(list(cls.builtin_function_guidance.keys()) + list(cls.external_function_guidance.keys())))

if __name__ == "__main__":
    function_manager = GUIFunctionManager()
    function_manager.add_gui_function(ExternalGUIFunction(function_name="test", callback=lambda: print("test"), function_args={}))