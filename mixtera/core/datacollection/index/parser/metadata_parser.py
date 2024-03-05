from abc import ABC, abstractmethod
from typing import Any, Optional

from mixtera.core.datacollection.index.index_collection import IndexFactory, IndexTypes, InMemoryDictionaryIndex


class MetadataParser(ABC):
    """
    Base class for parsing metadata. This class should be extended for parsing
    specific data collections' metadata. This class should be instantiated for
    each dataset and file, potentially allowing for parallel processing. When
    the object has completed its parsing job, the `mark_complete` method should
    be called. This compresses the index transparently.
    """

    def __init__(
        self, dataset_id: int, file_id: int, index_type: Optional[IndexTypes] = IndexTypes.IN_MEMORY_DICT_LINES
    ):
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
        self._parsing_complete = False
        self._index: InMemoryDictionaryIndex = IndexFactory.create_index(index_type)

    @abstractmethod
    def parse(self, line_number: int, payload: Any, **kwargs: Optional[dict[Any, Any]]) -> None:
        """
        Parses the given medata object and extends the given index in place.

        Args:
          line_number: the line number of the current instance
          payload: the metadata object to parse
          **kwargs: any other arbitrary keyword arguments required to parse metadata
        """
        raise NotImplementedError()

    def get_index(self) -> InMemoryDictionaryIndex:
        """
        Returns the fully parsed metadata index. This method should only be called
        once the index has been marked as complete.
        """
        assert self._parsing_complete, (
            "Retrieving index without first marking parsing as complete. " "Index will be transparently compressed!"
        )
        return self._index

    def mark_complete(self) -> None:
        """
        Compresses the underlying index, allowing it to be exposed for reading.
        """
        if not self._parsing_complete:
            self._parsing_complete = True
            self._index = self._index.compress()

    def is_parsing_complete(self) -> bool:
        """
        True if the parsing is complete, and the underlying index has been
        converted to ranges.
        """
        return self._parsing_complete
