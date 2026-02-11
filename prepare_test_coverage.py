"""
Prepare / update a pytest test environment for a Python project.

Recursively scans a source directory, analyses every .py file with the ast
module, and generates (or updates) matching test stubs under a test directory.

Usage:
    python prepare_test_coverage.py --src_path <source_dir> --test_path <test_dir>
"""

import argparse
import ast
import sys
import textwrap
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ── Helpers ──────────────────────────────────────────────────────────────────

# Built-in names we never want to mock (non-exhaustive but catches the common
# ones so generated stubs stay clean).
BUILTIN_NAMES: set[str] = {
    "abs",
    "all",
    "any",
    "ascii",
    "bin",
    "bool",
    "breakpoint",
    "bytearray",
    "bytes",
    "callable",
    "chr",
    "classmethod",
    "compile",
    "complex",
    "delattr",
    "dict",
    "dir",
    "divmod",
    "enumerate",
    "eval",
    "exec",
    "filter",
    "float",
    "format",
    "frozenset",
    "getattr",
    "globals",
    "hasattr",
    "hash",
    "help",
    "hex",
    "id",
    "input",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "locals",
    "map",
    "max",
    "memoryview",
    "min",
    "next",
    "object",
    "oct",
    "open",
    "ord",
    "pow",
    "print",
    "property",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "setattr",
    "slice",
    "sorted",
    "staticmethod",
    "str",
    "sum",
    "super",
    "tuple",
    "type",
    "vars",
    "zip",
}

TEXTUAL_BASES: set[str] = {"App", "Widget", "Screen", "Container", "TextArea", "Static"}


def default_value_for_type(type_str: Optional[str]) -> str:
    """Return a sensible default literal for a given type-annotation string."""
    if not type_str:
        return "None"
    t = type_str.strip().lower()
    if "int" in t:
        return "0"
    if "float" in t:
        return "0.0"
    if "str" in t:
        return '""'
    if "bool" in t:
        return "False"
    if "list" in t or "sequence" in t:
        return "[]"
    if "dict" in t or "mapping" in t:
        return "{}"
    if "set" in t:
        return "set()"
    if "tuple" in t:
        return "()"
    return "None"


def get_smart_default(arg_name: str, type_str: Optional[str]) -> str:
    """Return a more likely default based on argument name if type is missing."""
    if type_str:
        return default_value_for_type(type_str)

    low = arg_name.lower()
    if "volume" in low:
        return "0.8"
    if "sample_rate" in low or "rate" in low:
        return "44100"
    if "bpm" in low:
        return "120.0"
    if "device" in low:
        return "None"
    if "file" in low or "path" in low:
        return '""'
    if "size" in low or "count" in low or "index" in low:
        return "0"
    return "None"


def _read_source(path: Path) -> str:
    """Read a Python source file, trying several encodings."""
    for enc in ("utf-8", "utf-8-sig", "utf-16", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise RuntimeError(f"Cannot decode {path} with any known encoding")


# ── AST analysis ─────────────────────────────────────────────────────────────


@dataclass
class CallInfo:
    """A function/method call found inside a body."""

    name: str  # The simple name (e.g. "add" even for self.add)
    is_method: bool  # True when called as  self.xxx(…) or obj.xxx(…)


@dataclass
class FuncInfo:
    """Information about one function or method extracted from AST."""

    name: str
    lineno: int
    args: list[str]  # param names (without 'self')
    arg_types: dict[str, Optional[str]]  # param_name -> annotation str
    return_type: Optional[str]
    calls: list[CallInfo]
    is_method: bool = False
    class_name: Optional[str] = None


@dataclass
class ClassInfo:
    """Information about one class extracted from AST."""

    name: str
    lineno: int
    methods: list[FuncInfo] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)


@dataclass
class FileAnalysis:
    """Full analysis of a single source file."""

    functions: list[FuncInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    # Lookup:  func_name → return_type  (all funcs and methods in the file)
    return_type_lookup: dict[str, Optional[str]] = field(default_factory=dict)


class SourceAnalyzer(ast.NodeVisitor):
    """Walk a module AST and collect function / class information."""

    def __init__(self) -> None:
        self.result = FileAnalysis()
        self._current_class: Optional[ClassInfo] = None

    # ── visitors ──────────────────────────────────────────────────────────

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # pylint: disable=invalid-name
        """
        Extract metadata from a function or method definition.

        NOTE: This method MUST be named ``visit_FunctionDef`` (not snake_case)
        because ``ast.NodeVisitor`` dispatches by exact AST node class name.
        """

        fi = self._extract_func(node)
        if self._current_class is not None:
            fi.is_method = True
            fi.class_name = self._current_class.name
            self._current_class.methods.append(fi)
        else:
            self.result.functions.append(fi)
        self.result.return_type_lookup[fi.name] = fi.return_type
        # Don't recurse into nested defs / classes inside functions
        return

    visit_AsyncFunctionDef = visit_FunctionDef  # treat async the same way

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # pylint: disable=invalid-name
        """
        Extract metadata from a class definition.

        NOTE: Must be named ``visit_ClassDef`` — see visit_FunctionDef.
        """
        ci = ClassInfo(
            name=node.name,
            lineno=node.lineno,
            bases=[ast.unparse(b) for b in node.bases],
        )
        self._current_class = ci
        self.generic_visit(node)  # will call visit_FunctionDef for methods
        self._current_class = None
        self.result.classes.append(ci)

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _extract_func(node: ast.FunctionDef) -> FuncInfo:
        """Extract metadata from a single function/method AST node."""
        # Arguments (skip 'self' / 'cls')
        raw_args = [a.arg for a in node.args.args]
        args = [a for a in raw_args if a not in ("self", "cls")]

        # Argument type annotations
        arg_types: dict[str, Optional[str]] = {}
        for a in node.args.args:
            if a.arg in ("self", "cls"):
                continue
            arg_types[a.arg] = ast.unparse(a.annotation) if a.annotation else None

        # Return type
        ret = ast.unparse(node.returns) if node.returns else None

        # Calls inside the body
        calls: list[CallInfo] = []
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            if isinstance(child.func, ast.Name):
                calls.append(CallInfo(name=child.func.id, is_method=False))
            elif isinstance(child.func, ast.Attribute):
                calls.append(CallInfo(name=child.func.attr, is_method=True))

        return FuncInfo(
            name=node.name,
            lineno=node.lineno,
            args=args,
            arg_types=arg_types,
            return_type=ret,
            calls=calls,
        )


def analyse_file(source: str) -> FileAnalysis:
    """Parse *source* and return a FileAnalysis."""
    tree = ast.parse(source)
    analyzer = SourceAnalyzer()
    analyzer.visit(tree)
    return analyzer.result


# ── Stub generation ──────────────────────────────────────────────────────────


def _mock_lines(
    calls: list[CallInfo],
    return_type_lookup: dict[str, Optional[str]],
    indent: str,
) -> list[str]:
    """Generate commented-out mock setup lines for the calls in a function."""
    lines: list[str] = []
    seen: set[str] = set()
    for c in calls:
        if c.name in seen or c.name in BUILTIN_NAMES:
            continue
        seen.add(c.name)
        rtype = return_type_lookup.get(c.name)
        default = default_value_for_type(rtype)
        lines.append(f"{indent}# mock_{c.name} = MagicMock(return_value={default})")
    return lines


def _arg_setup_lines(
    fi: FuncInfo,
    indent: str,
) -> list[str]:
    """Generate variable assignments for function arguments with defaults."""
    lines: list[str] = []
    for arg in fi.args:
        ann = fi.arg_types.get(arg)
        val = get_smart_default(arg, ann)
        lines.append(f"{indent}{arg} = {val}")
    return lines


def generate_function_stub(
    fi: FuncInfo,
    module_import: str,
    lookup: dict[str, Optional[str]],
) -> str:
    """Generate a test stub for a standalone function."""
    indent = "    "
    lines: list[str] = []
    test_name = f"test_{fi.name}"
    lines.append(f"def {test_name}():")
    lines.append(f'{indent}"""Test for {fi.name}."""')

    # Setup section
    lines.append(f"{indent}# -- Setup --")
    lines.extend(_arg_setup_lines(fi, indent))
    lines.extend(_mock_lines(fi.calls, lookup, indent))

    # Call
    args_str = ", ".join(fi.args)
    lines.append(f"{indent}# -- Act --")
    lines.append(f"{indent}result = {module_import}.{fi.name}({args_str})")

    # Assert
    expected = default_value_for_type(fi.return_type)
    lines.append(f"{indent}# -- Assert --")
    lines.append(f"{indent}assert result == {expected}")

    # Special handling for main() and loops: comment them out by default
    loop_names = {"main", "run", "listen", "start"}
    if fi.name in loop_names or fi.name.endswith("_loop"):
        return "\n".join([f"# {ll}" for ll in lines])

    return "\n".join(lines)


def generate_class_stub(
    ci: ClassInfo,
    module_import: str,
    lookup: dict[str, Optional[str]],
) -> str:
    """Generate a test class stub for a source class."""
    indent = "    "
    indent2 = "        "
    indent3 = "            "
    indent4 = "                "
    lines: list[str] = []
    class_test_name = f"Test{ci.name}"

    is_textual_app = any(b == "App" for b in ci.bases)

    if is_textual_app:
        lines.append("@pytest.mark.asyncio")

    lines.append(f"class {class_test_name}:")
    lines.append(f'{indent}"""Tests for {ci.name}."""')
    lines.append("")

    if is_textual_app:
        # Specialized pilot test for Apps
        lines.append(f"{indent}async def test_app_run(self):")
        lines.append(f'{indent2}"""Test that the app starts and closes."""')
        lines.append(f"{indent2}# -- Setup --")
        init_info = next((m for m in ci.methods if m.name == "__init__"), None)
        init_args_str = ""
        if init_info:
            lines.extend(_arg_setup_lines(init_info, indent2))
            init_args_str = ", ".join(init_info.args)

        # Mock LiveSession if it's AlgoRaveApp to avoid audio thread blocking
        if ci.name == "AlgoRaveApp":
            lines.append(f'{indent2}with patch("{module_import}.LiveSession"):')
            lines.append(
                f"{indent3}application = {module_import}.{ci.name}({init_args_str})"
            )
            lines.append(f"{indent3}# -- Act & Assert --")
            lines.append(f"{indent3}async with application.run_test() as pilot:")
            lines.append(f"{indent4}# Simulate exit")
            lines.append(f'{indent4}await pilot.press("q")')
        else:
            lines.append(
                f"{indent2}application = {module_import}.{ci.name}({init_args_str})"
            )
            lines.append(f"{indent2}# -- Act & Assert --")
            lines.append(f"{indent2}async with application.run_test() as pilot:")
            lines.append(f"{indent3}# Simulate exit")
            lines.append(f'{indent3}await pilot.press("q")')
        return "\n".join(lines)

    # Standard setup_method
    lines.append(f"{indent}def setup_method(self):")
    lines.append(f'{indent2}"""Create a fresh instance for each test."""')

    # Find __init__ to get constructor arguments
    init_info = next((m for m in ci.methods if m.name == "__init__"), None)
    init_args_str = ""
    if init_info:
        lines.append(f"{indent2}# -- Setup Constructor Arguments --")
        lines.extend(_arg_setup_lines(init_info, indent2))
        init_args_str = ", ".join(init_info.args)

    lines.append(f"{indent2}self.instance = {module_import}.{ci.name}({init_args_str})")

    # Add teardown_method if class has a 'stop' or 'close' method
    has_cleanup = any(m.name in ("stop", "close") for m in ci.methods)
    if has_cleanup:
        cleanup_name = next(m.name for m in ci.methods if m.name in ("stop", "close"))
        lines.append("")
        lines.append(f"{indent}def teardown_method(self):")
        lines.append(f'{indent2}"""Clean up after each test."""')
        lines.append(f"{indent2}if hasattr(self, 'instance'):")
        lines.append(f"{indent3}self.instance.{cleanup_name}()")

    for mi in ci.methods:
        if mi.name.startswith("__") and mi.name.endswith("__"):
            # Skip dunder methods (e.g. __init__, __repr__) by default
            continue
        lines.append("")
        method_lines = _generate_method_lines(mi, ci.name, indent, indent2, lookup)
        lines.extend(method_lines)

    return "\n".join(lines)


# ── Test file management ─────────────────────────────────────────────────────


def _existing_test_names(content: str) -> set[str]:
    """Return the set of test-function / test-method names already present."""
    names: set[str] = set()
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return names
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                names.add(node.name)
    return names


def _existing_class_names(content: str) -> set[str]:
    """Return test class names already present in the test file."""
    names: set[str] = set()
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return names
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            names.add(node.name)
    return names


def _methods_in_test_class(content: str, class_name: str) -> set[str]:
    """Return test method names inside a specific test class."""
    names: set[str] = set()
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return names
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name.startswith("test_"):
                        names.add(item.name)
    return names


def _compute_module_import(src_file: Path, src_root: Path) -> str:
    """Compute the dotted module name for import statements.

    E.g.  src_root/sub/hello.py  →  'sub.hello'
    """
    rel = src_file.relative_to(src_root).with_suffix("")
    return ".".join(rel.parts)


def _build_import_header(module_import: str) -> str:
    """Return the header lines for a brand-new test file."""
    return textwrap.dedent(f"""\
        \"\"\"Auto-generated test stubs for {module_import}.\"\"\"

        import pytest
        from unittest.mock import MagicMock, patch
        import {module_import}
    """)


def _generate_method_lines(
    mi: FuncInfo,
    ci_name: str,
    indent: str,
    indent2: str,
    lookup: dict[str, Optional[str]],
) -> list[str]:
    """Generate the lines for a single method test."""
    lines: list[str] = []
    test_method_name = f"test_{mi.name}"
    lines.append(f"{indent}def {test_method_name}(self):")
    lines.append(f'{indent2}"""Test for {ci_name}.{mi.name}."""')

    # Setup
    lines.append(f"{indent2}# -- Setup --")
    lines.extend(_arg_setup_lines(mi, indent2))
    lines.extend(_mock_lines(mi.calls, lookup, indent2))

    # Call
    args_str = ", ".join(mi.args)
    lines.append(f"{indent2}# -- Act --")
    lines.append(f"{indent2}result = self.instance.{mi.name}({args_str})")

    # Assert
    expected = default_value_for_type(mi.return_type)
    lines.append(f"{indent2}# -- Assert --")
    lines.append(f"{indent2}assert result == {expected}")

    # Comment out loops
    loop_names = {"main", "run", "listen", "start"}
    if mi.name in loop_names or mi.name.endswith("_loop"):
        lines = [f"# {line}" if line.strip() else line for line in lines]

    return lines


def _generate_method_stub_for_existing_class(
    mi: FuncInfo,
    ci_name: str,
    lookup: dict[str, Optional[str]],
) -> str:
    """Generate a standalone method stub to append inside an existing test class."""
    lines = _generate_method_lines(mi, ci_name, "    ", "        ", lookup)
    return "\n".join(lines)


def _find_class_end_line(content: str, class_name: str) -> Optional[int]:
    """Find the last line number (1-indexed) of a class definition in content."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node.end_lineno
    return None


def process_file(
    src_file: Path,
    src_root: Path,
    test_root: Path,
) -> Optional[Path]:
    """Analyse one source file and create / update its test file.

    Returns the test-file path if it was written, else None.
    """
    # Skip __init__.py, setup.py, conftest.py etc.
    if src_file.name.startswith("__") or src_file.name in ("setup.py", "conftest.py"):
        return None

    # Read & parse source
    try:
        source = _read_source(src_file)
    except RuntimeError as exc:
        print(f"  ⚠  {exc}")
        return None

    try:
        analysis = analyse_file(source)
    except SyntaxError:
        print(f"  ⚠  Syntax error in {src_file}, skipping")
        return None

    if not analysis.functions and not analysis.classes:
        return None  # nothing to test

    # Determine test file path
    rel = src_file.relative_to(src_root)
    test_file = test_root / rel.parent / f"test_{src_file.name}"

    # Ensure directory exists + __init__.py files
    test_file.parent.mkdir(parents=True, exist_ok=True)
    for parent in test_file.relative_to(test_root).parents:
        init = test_root / parent / "__init__.py"
        if not init.exists():
            init.write_text("", encoding="utf-8")

    module_import = _compute_module_import(src_file, src_root)
    lookup = analysis.return_type_lookup

    # Read existing content or create header
    if test_file.exists():
        existing = _read_source(test_file)
    else:
        existing = ""

    existing_tests = _existing_test_names(existing)
    existing_classes = _existing_class_names(existing)

    # Accumulate new content to append
    new_blocks: list[str] = []

    # If file is brand new, add the header
    if not existing.strip():
        new_blocks.append(_build_import_header(module_import))

    # -- For lines to insert into existing test classes ---------------------
    # We track these separately because they need to be injected at specific
    # positions rather than appended at the end.
    class_injections: dict[str, list[str]] = {}  # test_class_name -> [stubs]

    # -- Generate stubs for top-level functions ----------------------------
    for fi in sorted(analysis.functions, key=lambda f: f.lineno):
        test_name = f"test_{fi.name}"
        if test_name in existing_tests:
            continue
        new_blocks.append(generate_function_stub(fi, module_import, lookup))

    # -- Generate stubs for classes ----------------------------------------
    for ci in sorted(analysis.classes, key=lambda c: c.lineno):
        test_class_name = f"Test{ci.name}"
        if test_class_name not in existing_classes:
            # Generate the entire class stub
            new_blocks.append(generate_class_stub(ci, module_import, lookup))
        else:
            # Class already exists — check for missing methods
            existing_methods = _methods_in_test_class(existing, test_class_name)
            for mi in ci.methods:
                if mi.name.startswith("__") and mi.name.endswith("__"):
                    continue
                test_method_name = f"test_{mi.name}"
                if test_method_name not in existing_methods:
                    stub = _generate_method_stub_for_existing_class(mi, ci.name, lookup)
                    class_injections.setdefault(test_class_name, []).append(stub)

    # -- Write results -----------------------------------------------------
    if not new_blocks and not class_injections:
        return None  # nothing new

    # Handle class injections (insert at the end of the existing class)
    output = existing
    for cls_name, stubs in class_injections.items():
        end_line = _find_class_end_line(output, cls_name)
        if end_line is not None:
            lines = output.splitlines(keepends=True)
            insert_pos = end_line  # insert after the last line of the class
            injection = "\n" + "\n\n".join(stubs) + "\n"
            lines.insert(insert_pos, injection)
            output = "".join(lines)

    # Append new blocks at the end
    if new_blocks:
        separator = "\n\n"
        appendix = separator + separator.join(new_blocks) + "\n"
        output = output.rstrip("\n") + "\n" + appendix

    test_file.write_text(output, encoding="utf-8")
    return test_file


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    """
    python prepare_test_coverage.py --src_path ./src --test_path ./tests
    """

    parser = argparse.ArgumentParser(
        description="Prepare / update pytest test stubs for a Python project.",
    )
    parser.add_argument(
        "--src_path",
        type=str,
        required=True,
        help="Root directory of the source code to analyse.",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        required=True,
        help="Root directory where test files will be created / updated.",
    )
    args = parser.parse_args()

    src_root = Path(args.src_path).resolve()
    test_root = Path(args.test_path).resolve()

    if not src_root.exists():
        print(f"Error: source directory '{src_root}' does not exist.")
        sys.exit(1)

    # Ensure test root exists
    test_root.mkdir(parents=True, exist_ok=True)

    # Generate / update conftest.py so pytest can import source modules
    conftest_path = test_root / "conftest.py"
    conftest_marker = "# -- auto-generated by prepare_test_coverage --"
    conftest_content = textwrap.dedent(f"""\
        {conftest_marker}
        import sys
        from pathlib import Path
        from unittest.mock import MagicMock

        # 1. Add source root to sys.path so imports like 'import sub.hello' work.
        _src_root = Path(r"{src_root}")
        if str(_src_root) not in sys.path:
            sys.path.insert(0, str(_src_root))

        # 2. Globally mock sounddevice to prevent CFFI errors and hardware access
        if "sounddevice" not in sys.modules:
            mock_sd = MagicMock()
            # Mock common classes/methods used in the project
            mock_sd.OutputStream.return_value = MagicMock()
            mock_sd.query_devices.return_value = []
            sys.modules["sounddevice"] = mock_sd
    """)
    if not conftest_path.exists():
        conftest_path.write_text(conftest_content, encoding="utf-8")
        print(f"  ✚  Created  conftest.py  (adds {src_root} to sys.path)")
    else:
        existing_conftest = _read_source(conftest_path)
        if conftest_marker not in existing_conftest:
            # Append our block so we don't overwrite user content
            with open(conftest_path, "a", encoding="utf-8") as f:
                f.write("\n\n" + conftest_content)
            print("  -  Updated  conftest.py  (appended sys.path setup)")

    # Collect python files
    py_files = sorted(src_root.rglob("*.py"))
    if not py_files:
        print(f"No .py files found under {src_root}")  # noqa: F541
        return

    print(f"Found {len(py_files)} Python file(s) under {src_root}\n")

    created = 0
    updated = 0
    for src_file in py_files:
        rel = src_file.relative_to(src_root)
        was_existing = (test_root / rel.parent / f"test_{src_file.name}").exists()
        result = process_file(src_file, src_root, test_root)
        if result is not None:
            if was_existing:
                updated += 1
                print(f"  ✏  Updated  {result.relative_to(test_root)}")
            else:
                created += 1
                print(f"  ✚  Created  {result.relative_to(test_root)}")
        else:
            print(f"  ·  Skipped  {rel}  (nothing new)")

    print(f"\nDone — {created} created, {updated} updated.")


if __name__ == "__main__":
    main()
