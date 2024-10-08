from abc import ABC, abstractmethod
from typing import Any, Optional

from loguru import logger


class MetadataParser(ABC):
    """
    Base class for parsing metadata. This class should be extended for parsing
    specific data collections' metadata. This class should be instantiated for
    each dataset and file, potentially allowing for parallel processing.
    """

    def __init__(self, dataset_id: int, file_id: int):
        """
        Initializes the metadata parser.

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
        Parses the given medata object and extends the internal state.

        Args:
          line_number: the line number of the current instance
          payload: the metadata object to parse
          **kwargs: any other arbitrary keyword arguments required to parse metadata
        """
        raise NotImplementedError()

    def add_metadata(self, sample_id: int, **kwargs: dict[str, Any]) -> None:
        internal_keys = set(["file_id", "dataset_id"])
        metadata = {"sample_id": sample_id}
        for key, value in kwargs.items():
            if key in internal_keys:
                logger.warning(f"You're supplying a Mixtera-internal key: {key}. Skipping.")
                continue

            # TODO(create issue): Allow non-list columns and enums
            metadata[key] = value if isinstance(value, list) else [value]
        self.metadata.append(metadata)
