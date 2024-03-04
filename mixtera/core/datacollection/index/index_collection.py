from collections import defaultdict
from copy import deepcopy
from enum import Enum
from typing import Union

from loguru import logger
from mixtera.core.datacollection.index import Index, IndexDatasetEntryType, IndexFeatureValueType, IndexType
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


class InMemoryDictionaryIndex(Index):
    """
    Represents an in memory dictionary class. This index exploits defaultdicts.
    """

    def __init__(self, pre_compressed: bool = False) -> None:
        """
        Initializes an `InMemoryDictionaryIndex` instance

        Args:
            pre_compressed: if False, this starts as a regular non-compressed index
            where scalar row indicators are added. `compress` needs to be called later
            to reduce the row indicators to compact row ranges. If True, this is
            readily an index where only row-ranges are allowed to be added.
        """
        self._is_compressed = pre_compressed
        self._index: defaultdict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    def compress(self) -> None:
        if not self._is_compressed:
            self._is_compressed = True
            for _0, feature_values in self._index.items():
                for _1, dataset_ids in feature_values.items():
                    for _2, file_ids in dataset_ids.items():
                        for file_key in file_ids.keys():
                            file_ids[file_key] = ranges(file_ids[file_key])

    def is_compressed(self) -> bool:
        return self._is_compressed

    def _optionally_compress(self) -> None:
        """
        Compresses the index if it is not already compressed, else NOOP.
        """
        if not self._is_compressed:
            self.compress()

    def get_full_index(self, copy: bool = False) -> IndexType:
        assert self._is_compressed, "You cannot access an uncompressed index!"
        return _return_with_copy_or_noop(self._index, copy)

    def get_by_feature(self, feature_name: str, copy: bool = False) -> IndexFeatureValueType:
        assert self._is_compressed, "You cannot access an uncompressed index!"
        return _return_with_copy_or_noop(self._index[feature_name], copy)

    def get_by_feature_value(
        self, feature_name: str, feature_value: Union[str, int, float], copy: bool = False
    ) -> IndexDatasetEntryType:
        assert self._is_compressed, "You cannot access an uncompressed index!"
        return _return_with_copy_or_noop(self._index[feature_name][feature_value], copy)

    def get_all_features(self) -> list[str]:
        assert self._is_compressed, "You cannot access an uncompressed index!"
        return list(self._index.keys())

    def merge(self, other: Index, copy_other: bool = False) -> None:
        assert self._is_compressed, "You cannot access an uncompressed index!"
        other_raw_dict = other.get_full_index(copy=copy_other)
        self._index = merge_dicts(self._index, other_raw_dict)

    def append_index_entry(
        self, feature_name: str, feature_value: Union[int, float, str], dataset_id: int, file_id: int, row_number: int
    ) -> None:
        if self._is_compressed:
            logger.warning(
                "Attempted addition to a closed and compressed index: "
                f"<{feature_name},{feature_value},{dataset_id},{file_id}> "
                "was not added!"
            )
            return
        self._index[feature_name][feature_value][dataset_id][file_id].append(row_number)

    def append_index_range(
        self,
        feature_name: str,
        feature_value: Union[int, float, str],
        dataset_id: int,
        file_id: int,
        low_bound: int,
        high_bound: int,
    ) -> None:
        assert self._is_compressed, "You can only add ranges to a compressed index!"
        self._index[feature_name][feature_value][dataset_id][file_id].append((low_bound, high_bound))

    def has_feature(self, feature_name: str) -> bool:
        return feature_name in self._index

    def keep_only_feature(self, feature_names: Union[str, list[str]]) -> None:
        if isinstance(feature_names, str):
            feature_names = [feature_names]
        feature_names = set(feature_names)
        to_delete = [key for key in self._index.keys() if key not in feature_names]
        for key in to_delete:
            del self._index[key]


class IndexTypes(Enum):
    """Contains the type of indexes supported by Mixtera"""

    IN_MEMORY_DICT_BASED = 1


class IndexFactory:
    @staticmethod
    def create_index(index_type: IndexTypes, **kwargs) -> Index:
        if index_type == IndexTypes.IN_MEMORY_DICT_BASED:
            return InMemoryDictionaryIndex(**kwargs)
        logger.error(f"Mixtera does not support index type {index_type}!")
        raise NotImplementedError()
