#
### Import Modules. ###
#
import os
import shutil
import inspect
#
import importlib.util
import importlib.machinery


#
def import_module_from_filepath(filepath: str, replace: list[tuple[str, str]] = []) -> object:

    """
    Imports a Python module from a filepath.

    Args:
        filepath (str): The path to the Python file to import as a module.

    Returns:
        object: The imported module object.
    """

    #
    base_filepath: str = ""

    #
    ### If asked to replace. ###
    #
    if replace:

        #
        ### Copy the base file to import into tmp.py. ###
        #
        base_filepath = filepath
        #
        filepath = "tmp.py"
        #
        with open(base_filepath, "r", encoding="utf-8") as f:
            #
            txt: str = f.read()
        #
        for rsrc, rdst in replace:
            #
            txt = txt.replace(rsrc, rdst)
        #
        print(f"\n\n\n```\n{txt}\n```\”\”\”")
        #
        with open(filepath, "w", encoding="utf-8") as f:
            #
            f.write(txt)

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
    ### If asked to replace. ###
    #
    if replace and base_filepath != "":
        #
        os.remove(filepath)
        #
        filepath = base_filepath

    #
    ### Return the imported module object. ###
    #
    return module

