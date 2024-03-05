from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from enum import Enum
from typing import Union

from loguru import logger
from mixtera.core.datacollection.index import (
    Index,
    IndexDatasetEntryRangeType,
    IndexFeatureValueRangeType,
    IndexRangeType,
)
from mixtera.utils import merge_dicts, ranges


def _return_with_copy_or_noop(to_return: Union[list, dict], copy: bool) -> Union[list, dict]:
    """
    This method either returns the passed object as is, or makes a deep copy
    of it, and returns that.

    Args:
      to_return: the object to be returned
      copy: whether to copy it or not

    Returns:
      The `to_return` object or a copy of it if `copy` is `True`
    """
    return to_return if not copy else deepcopy(to_return)


class InMemoryDictionaryIndex(Index, ABC):
    """
    Represents an in memory dictionary class. This index exploits defaultdicts.
    """

    def __init__(self) -> None:
        """
        Initializes an `InMemoryDictionaryIndex` instance

        Args:
            pre_compressed: if False, this starts as a regular non-compressed index
            where scalar row indicators are added. `compress` needs to be called later
            to reduce the row indicators to compact row ranges. If True, this is
            readily an index where only row-ranges are allowed to be added.
        """
        self._is_compressed = False
        self._index: defaultdict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    @abstractmethod
    def compress(self) -> "InMemoryDictionaryRangeIndex":
        """
        Compresses the internal index, reducing contiguous line ranges to spans.
        E.g. [1,2,3,5,6] --> [(1,4), (5,7)]. All modifications are done in place
        on the index. Note that the lower bound of each range is inclusive, but
        the upper bound is exclusive. This converts the index from the
        `IndexUncompressedType` to the `IndexType`
        """
        raise NotImplementedError("Method must be implemented in subclass!")

    def is_compressed(self) -> bool:
        return self._is_compressed

    def get_full_index(self, copy: bool = False) -> IndexRangeType:
        return _return_with_copy_or_noop(self._index, copy)

    def get_by_feature(self, feature_name: str, copy: bool = False) -> IndexFeatureValueRangeType:
        if feature_name not in self._index:
            return {}
        return _return_with_copy_or_noop(self._index[feature_name], copy)

    def get_by_feature_value(
        self, feature_name: str, feature_value: Union[str, int, float], copy: bool = False
    ) -> IndexDatasetEntryRangeType:
        if feature_name not in self._index or feature_value not in self._index[feature_name]:
            return {}
        return _return_with_copy_or_noop(self._index[feature_name][feature_value], copy)

    def get_all_features(self) -> list[str]:
        return list(self._index.keys())

    def merge(self, other: Index, copy_other: bool = False) -> None:
        assert isinstance(other, self.__class__), (
            "You cannot merge two indices of differnt types: "
            f"<left: {self.__class__}> and <right: {other.__class__}>"
        )
        other_raw_dict = other.get_full_index(copy=copy_other)
        self._index = merge_dicts(self._index, other_raw_dict)

    def has_feature(self, feature_name: str) -> bool:
        return feature_name in self._index

    def keep_only_feature(self, feature_names: Union[str, list[str]]) -> None:
        if isinstance(feature_names, str):
            feature_names = [feature_names]
        feature_names = set(feature_names)  # type: ignore[assignment]
        to_delete = [key for key in self._index.keys() if key not in feature_names]
        for key in to_delete:
            del self._index[key]


class InMemoryDictionaryRangeIndex(InMemoryDictionaryIndex):
    """
    In memory dictionary index that stores lists of ranges at the leaf nodes
    as opposed to lists of row indices.
    """

    def __init__(self) -> None:
        super().__init__()
        self._is_compressed = True

    def compress(self) -> "InMemoryDictionaryRangeIndex":
        """
        NOOP method, as an InMemoryDictionaryRangeIndex should always be compressed

        Returns:
            the self object
        """
        return self

    def append_entry(
        self,
        feature_name: str,
        feature_value: Union[int, float, str],
        dataset_id: int,
        file_id: int,
        payload: Union[int, tuple[int, int]],
    ) -> None:
        assert isinstance(payload, tuple), "InMemoryDictionaryRangeIndex can only append a range tuple!"
        self._index[feature_name][feature_value][dataset_id][file_id].append(payload)


class InMemoryDictionaryLineIndex(InMemoryDictionaryIndex):
    """
    In memory dictionary index that stores lists of row indices at the leaf
    nodes as opposed to lists of ranges.
    """

    def compress(self) -> "InMemoryDictionaryRangeIndex":
        compressed_index = InMemoryDictionaryRangeIndex()
        for feature_name, feature_values in self._index.items():
            for feature_value, dataset_ids in feature_values.items():
                for dataset_id, file_ids in dataset_ids.items():
                    for file_id, row_ids in file_ids.items():
                        compressed_index._index[feature_name][feature_value][dataset_id][file_id] = ranges(row_ids)
        return compressed_index

    def append_entry(
        self,
        feature_name: str,
        feature_value: Union[int, float, str],
        dataset_id: int,
        file_id: int,
        payload: Union[int, tuple[int, int]],
    ) -> None:
        assert isinstance(payload, int), "InMemoryDictionaryLineIndex can only append an int value!"
        self._index[feature_name][feature_value][dataset_id][file_id].append(payload)


class IndexTypes(Enum):
    """Contains the type of indexes supported by Mixtera"""

    IN_MEMORY_DICT_LINES = 1
    IN_MEMORY_DICT_RANGE = 2


class IndexFactory:
    @staticmethod
    def create_index(index_type: IndexTypes) -> InMemoryDictionaryIndex:
        if index_type == IndexTypes.IN_MEMORY_DICT_LINES:
            return InMemoryDictionaryLineIndex()
        if index_type == IndexTypes.IN_MEMORY_DICT_RANGE:
            return InMemoryDictionaryRangeIndex()
        logger.error(f"Mixtera does not support index type {index_type}!")
        raise NotImplementedError()
