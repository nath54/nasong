#
### Import Modules. ###
#
import os
import inspect
#
import importlib.util
import importlib.machinery


#
def import_module_from_filepath(filepath: str) -> object:

    """
    Imports a Python module from a filepath.

    Args:
        filepath (str): The path to the Python file to import as a module.

    Returns:
        object: The imported module object.
    """

    #
    ### Extract the module name from the filepath by removing the extension and path. ###
    #
    module_name: str = os.path.splitext(os.path.basename(filepath))[0]

    #
    ### Create a module specification using the module name and filepath, spec is of type ModuleSpec. ###
    #
    spec: Optional[importlib.machinery.ModuleSpec] = importlib.util.spec_from_file_location(module_name, filepath)

    #
    ### Check for errors. ###
    #
    if spec is None or spec.loader is None:
        #
        raise ImportError(f"Error : can't load module from file : {filepath}")

    #
    ### Create a module object from the specification, module type is dynamically determined so using Any. ###
    #
    module: Any = importlib.util.module_from_spec(spec)

    #
    ### Execute the module code in the module object's namespace, populating the module. ###
    #
    spec.loader.exec_module(module)

    #
    ### Return the imported module object. ###
    #
    return module

