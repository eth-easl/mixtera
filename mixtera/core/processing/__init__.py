"""
This submodule contains code for executing pre-query operations
"""

import os

from .execution_mode import ExecutionMode  # noqa: F401

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
