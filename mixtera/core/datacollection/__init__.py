"""
This submodule contains Mixtera's datasets, which are the main API of Mixtera
"""

from collections import defaultdict
from typing import Union

IndexType = defaultdict[str, defaultdict[str, defaultdict[int, dict[int, list[tuple[int, int]]]]]]
UncompressedIndexType = defaultdict[
    str, defaultdict[str, defaultdict[int, dict[int, list[Union[int, tuple[int, int]]]]]]
]


from .data_collection import MixteraDataCollection  # noqa: F401,E402 # pylint: disable=wrong-import-position
from .property import Property  # noqa: F401,E402 # pylint: disable=wrong-import-position
from .property_type import PropertyType  # noqa: F401,E402 # pylint: disable=wrong-import-position

__all__ = ["MixteraDataCollection", "Property", "PropertyType", "IndexType", "UncompressedIndexType"]
