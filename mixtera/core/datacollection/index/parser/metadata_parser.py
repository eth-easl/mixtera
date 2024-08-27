from abc import ABC, abstractmethod
from typing import Any, Optional


class MetadataParser(ABC):
    """
    Base class for parsing metadata. This class should be extended for parsing
    specific data collections' metadata. This class should be instantiated for
    each dataset and file, potentially allowing for parallel processing. When
    the object has completed its parsing job, the `mark_complete` method should
    be called. This compresses the index transparently.

    TODO (MaxiBoether): adjust docstring
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
        self.metadata: list[dict] = []

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

    def add_metadata(self, sample_id: int, **kwargs: dict[str, Any]) -> None:
        metadata = {"sample_id": sample_id}
        for key, value in kwargs.items():
            metadata[key] = value if isinstance(value, list) else [value]
        self.metadata.append(metadata)
