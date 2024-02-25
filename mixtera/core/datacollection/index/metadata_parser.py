from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Optional

from mixtera.core.datacollection import IndexType, UncompressedIndexType


class MetadataParser(ABC):
    """
    Base class for parsing metadata. This class should be extended for parsing
    specific data collections' metadata. This class should be instantiated for
    each dataset and file, potentially allowing for parallel processing.
    """

    def __init__(self, dataset_id: int, file_id: int):
        """
        Initializes the metadata parser. This initializer also sets up its own
        index structure that gets manipulated. In the future, the index structure
        might be passed as a parameter to the initializer.

        Args:
          dataset_id: the id of the source dataset
          file_id: the id of the source file
        """
        self.dataset_id: int = dataset_id
        self.file_id: int = file_id
        # self._index = UncompressedIndexType()
        self._index: UncompressedIndexType = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )

    @abstractmethod
    def parse(self, line_number: int, metadata: Any, **kwargs: Optional[dict[Any, Any]]) -> None:
        """
        Parses the given medata object and extends the given index in place.

        Args:
          line_number: the line number of the current instance
          metadata: the metadata object to parse
          **kwargs: any other arbitrary keyword arguments required to parse metadata
        """
        raise NotImplementedError()

    @abstractmethod
    def get_index(self) -> IndexType:
        """
        Returns the fully parsed metadata index
        """
        raise NotImplementedError()
