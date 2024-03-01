from abc import ABC, abstractmethod
from typing import Union

from mixtera.core.datacollection.index import IndexFeatureValueType, \
  IndexDatasetEntryType


class IndexBase(ABC):
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
  """

  @abstractmethod
  def append_index_entry(self, feature_name: str, feature_value: Union[int, float, str],
                         dataset_id: int, file_id: int, row_number: int) -> None:
    """
    Appends a new row number entry to the index.

    Args:
      feature_name: the name of feature (e.g. 'language')
      feature_value: the value of the feature (e.g. 'Italian')
      dataset_id: the id of the dataset
      file_id: the id of the file within the dataset
      row_number: the row number of the valid instance
    """
    raise NotImplementedError("Method must be implemented in subclass!")

  @abstractmethod
  def get_by_feature(self, feature_name: str, copy=False) -> IndexFeatureValueType:
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
  def get_by_feature_value(self, feature_name: str, feature_value: Union[str, int, float],
                           copy=False) -> IndexDatasetEntryType:
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
  def merge(self, other: "IndexBase") -> None:
    """
    Merges another index into this one.

    Args:
      other: The other index

    Returns:
      Does not return anything, but extends the internal index with the `other`.
    """
    raise NotImplementedError("Method must be implemented in subclass!")
