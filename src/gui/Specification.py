from typing import *
import inspect

class ArgumentSpecification: 
    def __init__(self, parameter_label: str, parameter_name: str, data_type: int | float | str | bool, default_value: Any):
        self.parameter_label = parameter_label
        self.parameter_name = parameter_name
        self.data_type = data_type
        self.default_value = default_value

class FunctionSpecification: 
    def __init__(self, function: Callable):
        self.function = function
        self.label = function.label # type: ignore
        self.name = function.__name__
        self.arg_specs: List[ArgumentSpecification] = []
        
        signature = inspect.signature(function)
        for i, (name, param) in enumerate(signature.parameters.items()):
            annotation = param.annotation if param.annotation is not inspect._empty else Any
            default = param.default if param.default is not inspect._empty else None

            arg_spec = ArgumentSpecification(
                parameter_label=function.parameter_labels[i], # type: ignore
                parameter_name=name,
                data_type=annotation, # type: ignore
                default_value=default
            )
            self.arg_specs.append(arg_spec)