"""
This submodule contains Mixtera's datasets, which are the main API of Mixtera
"""

import os
from collections import defaultdict

IndexType = defaultdict[str, defaultdict[str, defaultdict[int, dict[int, list[tuple[int, int]]]]]]


from .data_collection import MixteraDataCollection  # noqa: F401,E402 # pylint: disable=wrong-import-position
from .property import Property  # noqa: F401,E402 # pylint: disable=wrong-import-position
from .property_type import PropertyType  # noqa: F401,E402 # pylint: disable=wrong-import-position

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
