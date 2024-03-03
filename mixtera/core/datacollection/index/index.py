from abc import ABC, abstractmethod
from typing import Union

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


class Index(ABC):
    """
    Abstract class that represents the interface of a Mixtera index.

    For now, it is assumed that there are 4 levels to an index:
    {
      "feature_name": {
        "feature_value": {
          dataset_id: {
            file_id: [
              line number | line range tuple
            ]
          }
        }
      }
    }

    Once the index is fully built, `compress()` should be called. This switches
    the representation internal representation from `IndexUncompressedType` to
    `IndexType`. No additions should be made to an index that has been
    finalized. The only changes that can be made refer to merging additional
    indexes into this one.
    """

    @abstractmethod
    def append_index_entry(
        self, feature_name: str, feature_value: Union[int, float, str], dataset_id: int, file_id: int, row_number: int
    ) -> None:
        """
        Appends a new row number entry to the index. For efficiency reasons, it
        is assumed for now that row numbers pertaining to the same feature, value,
        dataset and file are always added in monotonically increasing order.

        Args:
          feature_name: the name of feature (e.g. 'language')
          feature_value: the value of the feature (e.g. 'Italian')
          dataset_id: the id of the dataset
          file_id: the id of the file within the dataset
          row_number: the row number of the valid instance
        """
        raise NotImplementedError("Method must be implemented in subclass!")

    @abstractmethod
    def get_full_index(self, copy: bool = False) -> IndexType:
        """
        Return the full index. The index should always be compressed into ranges.

        Args:
          copy: if True, returns a copy of the index. Otherwise, it returns the
          internal index, which can lead to side-effects.

        Returns:
          An `IndexType` object.
        """
        raise NotImplementedError("Method must be implemented in subclass!")

    @abstractmethod
    def get_by_feature(self, feature_name: str, copy: bool = False) -> "IndexFeatureValueType":
        """
        Returns the entries under the name of this feature

        Args:
          feature_name: the name of the feature
          copy: if True, the returned dictionary is a copy of the internal data,
            meaning no side-effects can arise by changing the returned data
            structure. This is more expensive, and deactivated by default.

        Returns:
          An instance of IndexFeatureValueType; if no such feature is found an
          empty dictionary is returned
        """
        raise NotImplementedError("Method must be implemented in subclass!")

    @abstractmethod
    def get_by_feature_value(
        self, feature_name: str, feature_value: Union[str, int, float], copy: bool = False
    ) -> IndexDatasetEntryType:
        """
        Returns the entries in the index for this feature and its value.

        Args:
          feature_name: the name of the feature
          feature_value: the value of the feature
          copy: if True, the returned dictionary is a copy of the internal data,
            meaning no side-effects can arise by changing the returned data
            structure. This is more expensive, and deactivated by default.

        Returns:
          An instance of IndexDatasetEntryType; if no such feature is found, or no
          such value exists, an empty dictionary is returned
        """
        raise NotImplementedError("Method must be implemented in subclass!")

    @abstractmethod
    def merge(self, other: "Index", copy_other: bool = False) -> None:
        """
        Merges another index into this one. It is assumed that the indexes never
        have collisions. In other words, they may map over the same feature, value,
        and dataset, but never over the same files within each category.

        Args:
          other: The other index
          copy_other: if True, the other dictionary is copied to avoid side-effects
            later due to shared pointers. If False, side-effects may exist, but
            merging is faster.

        Returns:
          Does not return anything, but extends the internal index with the `other`.
        """
        raise NotImplementedError("Method must be implemented in subclass!")

    @abstractmethod
    def compress(self) -> None:
        """
        Compresses the internal index, reducing contiguous line ranges to spans.
        E.g. [1,2,3,5,6] --> [(1,4), (5,7)]. All modifications are done in place
        on the index. Note that the lower bound of each range is inclusive, but
        the upper bound is exclusive. This converts the index from the
        `IndexUncompressedType` to the `IndexType`
        """
        raise NotImplementedError("Method must be implemented in subclass!")

    @abstractmethod
    def is_compressed(self) -> bool:
        """
        Returns True if the index is compressed and safe to read, otherwise False.
        """
        raise NotImplementedError("Method must be implemented in subclass!")

    @abstractmethod
    def get_all_features(self) -> list[str]:
        """
        Returns all top level keys (i.e. feature names).

        Returns:
            A list of keys to access the first level of the index via the
            `get_by_feature` method
        """
        raise NotImplementedError("Method must be implemented in subclass!")
