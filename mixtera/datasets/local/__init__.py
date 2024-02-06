"""
This submodule contains Mixtera's local dataset
"""

import os

from .local_mixtera_dataset import LocalMixteraDataset  # noqa: F401

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
