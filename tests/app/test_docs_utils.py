


"""Auto-generated test stubs for app.docs_utils."""

import pytest
from unittest.mock import MagicMock, patch
import app.docs_utils


def test_get_module_docs():
    """Test for get_module_docs."""
    # -- Setup --
    package_name = ""
    # mock_getmembers = MagicMock(return_value=None)
    # mock_import_module = MagicMock(return_value=None)
    # mock_iter_modules = MagicMock(return_value=None)
    # mock_isclass = MagicMock(return_value=None)
    # mock_get_module_docs = MagicMock(return_value={})
    # mock_getdoc = MagicMock(return_value=None)
    # mock_isfunction = MagicMock(return_value=None)
    # -- Act --
    result = app.docs_utils.get_module_docs(package_name)
    # -- Assert --
    assert result == {}

def test_flatten_docs():
    """Test for flatten_docs."""
    # -- Setup --
    docs = {}
    prefix = ""
    # mock_items = MagicMock(return_value=None)
    # mock_append = MagicMock(return_value=None)
    # mock_extend = MagicMock(return_value=None)
    # mock_get = MagicMock(return_value=None)
    # mock_flatten_docs = MagicMock(return_value=[])
    # -- Act --
    result = app.docs_utils.flatten_docs(docs, prefix)
    # -- Assert --
    assert result == []
