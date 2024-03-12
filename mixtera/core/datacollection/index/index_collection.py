from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import Callable, Union

from loguru import logger
from mixtera.core.datacollection.index import (
    Index,
    IndexDatasetEntryRangeType,
    IndexFeatureValueType,
    IndexRangeType,
    IndexType,
)
from mixtera.utils import merge_dicts, ranges
from mixtera.utils.utils import return_with_deepcopy_or_noop


def raw_index_dict_instantiator() -> IndexType:
    """
    Instantiates and returns a raw index dict of `IndexType`.
    """
    return defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))


class InMemoryDictionaryIndex(Index, ABC):
    """
    Represents a base class for in memory dictionary indexes. This class should
    not be used directly, but rather should act as a type. Concrete indexes
    should extend this class and implement its abstract methods.
    """

    def __init__(self) -> None:
        """
        Based initializer for an `InMemoryDictionaryIndex` instance
        """
        self._is_compressed = False
        self._index: defaultdict = raw_index_dict_instantiator()

    def _copy_constructor(self, index_payload: IndexType) -> "InMemoryDictionaryIndex":
        """
        Creates a new InMemoryDictionaryIndex object from the given `index_payload` dict.
        It should be noted that the concrete type of the new object will be one of the
        subclasses' (as only they can be instantiated).

        Args:
            index_payload: the payload from which the new index object should be created

        Returns:
            A new index object with the same type as the calling object.
        """
        new_index = self.__class__()
        new_index._index = index_payload
        return new_index

    @abstractmethod
    def compress(self) -> "InMemoryDictionaryRangeIndex":
        """
        Compresses the internal index, reducing contiguous line ranges to spans.
        E.g. [1,2,3,5,6] --> [(1,4), (5,7)]. Note that the lower bound of each
        range is inclusive, but the upper bound is exclusive.

        Returns:
          An index of the `InMemoryDictionaryRangeIndex` type.
        """
        raise NotImplementedError("Method must be implemented in subclass!")

    @property
    def is_compressed(self) -> bool:
        return self._is_compressed

    def get_full_dict_index(self, copy: bool = False) -> IndexRangeType:
        return return_with_deepcopy_or_noop(self._index, copy)

    def get_index_by_features(self, feature_names: Union[str, list[str]], copy: bool = False) -> Index:
        return self._copy_constructor(self.get_dict_index_by_many_features(feature_names, copy=copy))

    def get_index_by_predicate(
        self, feature_name: str, predicate: Callable[[Union[str, int, float]], bool], copy: bool = False
    ) -> Index:
        result = self.get_dict_index_by_predicate(feature_name, predicate, copy=copy)
        if result:
            result = {feature_name: self.get_dict_index_by_predicate(feature_name, predicate, copy=copy)}
        return self._copy_constructor(result)

    def get_dict_index_by_many_features(self, feature_names: Union[str, list[str]], copy: bool = False) -> IndexType:
        if isinstance(feature_names, str):
            feature_names = [feature_names]

        result = raw_index_dict_instantiator()
        for feature_name in feature_names:
            if feature_name in self._index:
                result[feature_name] = self._index[feature_name]

        if not result:
            logger.warning(f"None of the features {feature_names} were found; returning empty result!")
            return result
        return return_with_deepcopy_or_noop(result, copy)

    def get_dict_index_by_feature(self, feature_name: str, copy: bool = False) -> IndexFeatureValueType:
        returned = self.get_dict_index_by_many_features(feature_name, copy)
        return returned[feature_name]

    def get_dict_index_by_feature_value(
        self, feature_name: str, feature_value: Union[str, int, float], copy: bool = False
    ) -> IndexDatasetEntryRangeType:
        if feature_name not in self._index or feature_value not in self._index[feature_name]:
            logger.warning(
                f"The feature {feature_name} or the feature value {feature_value} was not "
                "found in index; returning emtpy dict!"
            )
            return {}
        return return_with_deepcopy_or_noop(self._index[feature_name][feature_value], copy)

    def get_dict_index_by_predicate(
        self, feature_name: str, predicate: Callable[[Union[str, int, float]], bool], copy: bool = False
    ) -> IndexFeatureValueType:
        if feature_name not in self._index:
            logger.warning(f"The feature {feature_name} was not found in index; returning emtpy dict!")
            return {}
        result = {k: v for k, v in self._index[feature_name].items() if predicate(k)}
        return return_with_deepcopy_or_noop(result, copy)

    def get_all_features(self) -> list[str]:
        return list(self._index.keys())

    def merge(self, other: Index, copy_other: bool = False) -> None:
        assert isinstance(other, self.__class__), (
            "You cannot merge two indices of differnt types: "
            f"<left: {self.__class__}> and <right: {other.__class__}>"
        )
        other_raw_dict = other.get_full_dict_index(copy=copy_other)
        self._index = merge_dicts(self._index, other_raw_dict)

    def has_feature(self, feature_name: str) -> bool:
        return feature_name in self._index

    def drop_other_features(self, feature_names: Union[str, list[str]]) -> None:
        if isinstance(feature_names, str):
            feature_names = [feature_names]
        feature_names = set(feature_names)  # type: ignore[assignment]
        to_delete = [key for key in self._index.keys() if key not in feature_names]
        for key in to_delete:
            del self._index[key]


class InMemoryDictionaryRangeIndex(InMemoryDictionaryIndex):
    """
    Concrete in memory dictionary index class that stores lists of ranges at
    the leaf nodes as opposed to lists of row indices.
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

    @property
    def values_count(self) -> int:
        total_count = 0
        for _, feature_values in self._index.items():
            for _, dataset_ids in feature_values.items():
                for _, file_ids in dataset_ids.items():
                    for _, file_ranges in file_ids.items():
                        for start, end in file_ranges:
                            total_count += end - start
        return total_count


class InMemoryDictionaryLineIndex(InMemoryDictionaryIndex):
    """
    Concrete in memory dictionary index that stores lists of row indices at
    the leaf nodes as opposed to lists of ranges.
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
