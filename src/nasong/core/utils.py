# Copyright (C) 2026 Nathan Cerisara <https://github.com/nath54/nasong>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


"""
Core utilities for the NaSong engine.

This module provides utility functions for dynamic module loading and other
common tasks used throughout the core library.
"""

#
### Import Modules. ###
#
from typing import Optional, Any

#
import os

#
import importlib.util
import importlib.machinery


#
def import_module_from_filepath(
    filepath: str, replace: list[tuple[str, str]] | None = None
) -> object:
    """Imports a Python module from a given filesystem path.

    This utility allows for dynamic loading of instruments or song definitions.
    It optionally supports on-the-fly string replacement in the source code
    before importing, which is primarily used for testing or hot-patching.

    Args:
        filepath (str): The absolute or relative path to the `.py` file.
        replace (list[tuple[str, str]], optional): A list of (old, new) string
            paires to replace in the source code before execution.
            Defaults to None.

    Returns:
        object: The imported module object.

    Raises:
        ImportError: If the module specification or loader cannot be created.
    """

    #
    base_filepath: str = ""

    #
    ### If asked to replace. ###
    #
    if replace is not None:
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
        print(f"\n\n\n```\n{txt}\n```\n\n\n")
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
    spec: Optional[importlib.machinery.ModuleSpec] = (
        importlib.util.spec_from_file_location(module_name, filepath)
    )

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
