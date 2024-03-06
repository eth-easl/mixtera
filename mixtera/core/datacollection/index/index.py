from abc import ABC, abstractmethod
from typing import Union

# Compressed hierarchy (i.e. uses ranges)
IndexRowRangeType = list[tuple[int, int]]
IndexFileEntryRangeType = dict[int, IndexRowRangeType]
IndexDatasetEntryRangeType = dict[int, IndexFileEntryRangeType]
IndexFeatureValueRangeType = dict[Union[int, float, str], IndexDatasetEntryRangeType]
IndexRangeType = dict[str, IndexFeatureValueRangeType]

# Uncompressed hierarchy (i.e. uses raw row identifiers)
IndexRowIndicatorsType = list[int]
IndexFileEntryLineType = dict[int, IndexRowIndicatorsType]
IndexDatasetEntryLineType = dict[int, IndexFileEntryLineType]
IndexFeatureValueLineType = dict[Union[int, float, str], IndexDatasetEntryLineType]
IndexLineType = dict[str, IndexFeatureValueLineType]


# Common types
IndexCommonType = IndexRowRangeType | IndexRowIndicatorsType
IndexFileEntryType = dict[int, IndexCommonType]
IndexDatasetEntryType = dict[int, IndexFileEntryType]
IndexFeatureValueType = dict[Union[int, float, str], IndexDatasetEntryType]
IndexType = dict[str, IndexFeatureValueType]


class Index(ABC):
    """
    Abstract class that represents the interface of a Mixtera index.

    For now, it is assumed that there are 4 levels to an index:
    {
      "feature_name": {
        "feature_value": {
          dataset_id: {
            file_id: [
              payload (e.g. could be row index or row range)
            ]
          }
        }
      }
    }
    """

    @abstractmethod
    def append_entry(
        self,
        feature_name: str,
        feature_value: Union[int, float, str],
        dataset_id: int,
        file_id: int,
        payload: Union[int, tuple[int, int]],
    ) -> None:
        """
        Appends a new payload entry to the index. For efficiency reasons, it
        is assumed that if the payloads are comparable, they are always added
        to the index in monotonically increasing order.

        Args:
          feature_name: the name of feature (e.g. 'language')
          feature_value: the value of the feature (e.g. 'Italian')
          dataset_id: the id of the dataset
          file_id: the id of the file within the dataset
          payload: the element to be added to the index (e.g. row range). The
            type of this element is ultimately determined by the implementing
            index class.
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
        have collisions. In other words, they may map over the same feature,
        value, and dataset, but never over the same files within each category.

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
    def get_all_features(self) -> list[str]:
        """
        Returns all top level keys (i.e. feature names).

        Returns:
            A list of keys to access the first level of the index via the
            `get_by_feature` method
        """
        raise NotImplementedError("Method must be implemented in subclass!")

    @abstractmethod
    def has_feature(self, feature_name: str) -> bool:
        """
        Checks if `feature_name` exists in the first level of the index.

        Args:
            feature_name: the name of the feature

        Returns:
            True if the feature exists in the index, False otherwise.
        """
        raise NotImplementedError("Method must be implemented in subclass!")

    @abstractmethod
    def keep_only_feature(self, feature_names: Union[str, list[str]]) -> None:
        """
        Discards all features that are not present in the `feature_names` parameter.

        Args:
            feature_names: can either be a list or a string literal. Indicates
            which features to keep and which to discard.
        """
        raise NotImplementedError("Method must be implemented in subclass!")
