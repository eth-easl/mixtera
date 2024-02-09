"""
This submodule contains general utility functions
"""

import os

from .utils import flatten, ranges, dict_into_dict, defaultdict_to_dict # noqa: F401

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
