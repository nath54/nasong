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
Documentation and introspection utilities for the NaSong application.

This module provides functions to recursively inspect NaSong packages and extract
docstrings for classes and functions, allowing for dynamic documentation generation
within the TUI/DAW.
"""

#
### Import Modules. ###
#
from typing import Dict, List, Any

#
import inspect
import pkgutil
import importlib


def get_module_docs(package_name: str) -> Dict[str, Any]:
    """
    Recursively inspects a package and returns its documentation structure.

    Args:
        package_name: The fully qualified name of the package/module to inspect.

    Returns:
        A dictionary containing:
            - "classes": Maps class names to their docstrings.
            - "functions": Maps function names to their docstrings.
            - "submodules": Recursively holds documentation for child modules.
    """
    results = {"classes": {}, "functions": {}, "submodules": {}}

    try:
        package = importlib.import_module(package_name)
    except ImportError:
        return results

    # Inspect current module
    for name, obj in inspect.getmembers(package):
        if inspect.isclass(obj) and obj.__module__ == package_name:
            results["classes"][name] = inspect.getdoc(obj) or "No description."
        elif inspect.isfunction(obj) and obj.__module__ == package_name:
            results["functions"][name] = inspect.getdoc(obj) or "No description."

    # Recurse into submodules
    if hasattr(package, "__path__"):
        for _, name, _is_pkg in pkgutil.iter_modules(package.__path__):
            full_name = f"{package_name}.{name}"
            results["submodules"][name] = get_module_docs(full_name)

    return results


def flatten_docs(docs: Dict, prefix: str = "") -> List[tuple]:
    """
    Flattens the hierarchical documentation structure for linear display (e.g., in a tree).

    Args:
        docs: The hierarchical documentation dictionary from `get_module_docs`.
        prefix: Current module path prefix for recursion.

    Returns:
        A list of tuples: (type, name, path, docstring).
    """
    items = []

    for name, doc in docs.get("classes", {}).items():
        items.append(("class", name, f"{prefix}", doc))

    for name, doc in docs.get("functions", {}).items():
        items.append(("function", name, f"{prefix}", doc))

    for name, sub_docs in docs.get("submodules", {}).items():
        new_prefix = f"{prefix}.{name}" if prefix else name
        items.extend(flatten_docs(sub_docs, new_prefix))

    return items
