from Option import Option, ComputableOption

def function_label(label: str):
    """
    Set a GUI-ready function label.

    See below for usage
    """
    def decorator(function):
        function.label = label
        return function
    return decorator

def param_labels(labels: list[str]):
    """
    Conveniently set the GUI-ready parameter labels corresponding to a function's parameters.
    The order the labels must match the order of the function's parameters.

    See below for usage.
    """
    def decorator(function):
        function.parameter_labels = labels
        return function
    return decorator

def options_provided(options: list[Option]):
    """
    Set the options for a parmeter. 

    len(options) must equal len(param_labels). 

    Example usage: 
    param_labels = ["name", "day"]
    options_provided = [
        None,                    # Denoting that this paraemter does not come with options
        ["Monday", "Wednesday"]  # Denoting that these are the options for `day`
    ]
    """
    def decorator(function):
        function.options_provided = options
        return function
    return decorator