"""
This submodule contains Mixtera's remote dataset
"""

import os

from .remote_mixtera_dataset import RemoteMixteraDataset  # noqa: F401

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
