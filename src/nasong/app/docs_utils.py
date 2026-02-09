import inspect
import pkgutil
import importlib
from typing import Dict, List, Any


def get_module_docs(package_name: str) -> Dict[str, Any]:
    """
    Recursively inspects a package and returns a dictionary structure:
    {
        "module_name": {
            "classes": { "ClassName": "docstring" },
            "functions": { "func_name": "docstring" },
            "submodules": { ... }
        }
    }
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
        for _, name, is_pkg in pkgutil.iter_modules(package.__path__):
            full_name = f"{package_name}.{name}"
            results["submodules"][name] = get_module_docs(full_name)

    return results


def flatten_docs(docs: Dict, prefix: str = "") -> List[tuple]:
    """
    Flattens the docs structure for easier tree population.
    Returns list of (type, name, path, docstring).
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
