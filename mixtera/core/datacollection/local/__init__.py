"""
This submodule contains Mixtera's local collection
"""

import os

from .local_collection import LocalDataCollection  # noqa: F401

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
