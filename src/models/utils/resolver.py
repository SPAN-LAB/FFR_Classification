import importlib
import inspect
import pkgutil

from ...models.utils import ModelInterface
# from  import Jason_CNN

def _iter_model_classes():
    """
    Yield ``(module_name, class_obj)`` for concrete ``ModelInterface`` implementations.
    """
    # Where all the models are found
    package_name = "src.models"

    # Finds all modules in src.models
    package = importlib.import_module(package_name)

    # Iterates over all the submodules, skipping any that are packages (i.e. files)
    for _, module_name, is_pkg in pkgutil.iter_modules(package.__path__):
        if is_pkg or module_name == "utils":
            continue
        
        # A module in src.models (a `.py` file)
        module = importlib.import_module(f"{package_name}.{module_name}")

        # Iterates over all the classes inside the module 
        for _, cls in inspect.getmembers(module, inspect.isclass):
            # Skips this class if it is not a subclass of `ModelInterface`
            # OR if it is the `ModelInterface` itself
            if not issubclass(cls, ModelInterface) or cls is ModelInterface:
                continue

            yield module_name, cls

def find_model(name: str) -> type[ModelInterface]:
    """
    Return the concrete class associated with ``name``.

    :param str name: the name of the file that contains the model implementation
    :raises ValueError: If no implementation matches ``name`` or if it is empty
    :returns: the type of the concrete class
    """
    if not name:
        raise ValueError("Model name must be a non-empty string.")

    name = name.lower()

    for module_name, cls in _iter_model_classes():
        if module_name.lower() == name:
            print("Found a module!")
            return cls

    raise ValueError(f"No model found matching '{name}'.")

def find_models() -> dict[str, type[ModelInterface]]:
    """
    Returns a dictionary that maps from the names of available models to 
    the model types.

    :returns: TODO
    """
    models = [(module_name, cls) for module_name, cls in _iter_model_classes()]
    map = {}
    for (name, cls) in models:
        map[name] = cls
    return map