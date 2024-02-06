"""
This submodule contains Mixtera's datasets, which are the main API of Mixtera
"""

import os

from .dataset_types import DatasetTypes  # noqa: F401
from .mixtera_dataset import MixteraDataset  # noqa: F401

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
