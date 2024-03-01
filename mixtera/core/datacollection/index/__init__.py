"""
This submodule contains implementations for Mixtera indexes
"""
from typing import Union

# The raw index types that are returned by the Index data structures

# Compressed hierarchy (i.e. uses ranges)
IndexRowRangeType = list[tuple[int, int]]
IndexFileEntryType = dict[int, IndexRowRangeType]
IndexDatasetEntryType = dict[int, IndexFileEntryType]
IndexFeatureValueType = dict[Union[int, float, str], IndexDatasetEntryType]
IndexType = dict[str, IndexFeatureValueType]

# Uncompressed hierarchy (i.e. uses raw row identifiers)
IndexRowIndicatorsType = list[int]
IndexFileEntryUncompressedType = dict[int, IndexRowIndicatorsType]
IndexDatasetEntryUncompressedType = dict[int, IndexFileEntryUncompressedType]
IndexFeatureValueUncompressedType = dict[Union[int, float, str], IndexDatasetEntryUncompressedType]
IndexUncompressedType = dict[str, IndexFeatureValueUncompressedType]


__all__ = [
  "IndexRowRangeType",
  "IndexFileEntryType",
  "IndexDatasetEntryType",
  "IndexFeatureValueType",
  "IndexType",

  "IndexRowIndicatorsType",
  "IndexFileEntryUncompressedType",
  "IndexDatasetEntryUncompressedType",
  "IndexFeatureValueUncompressedType",
  "IndexUncompressedType",
]
