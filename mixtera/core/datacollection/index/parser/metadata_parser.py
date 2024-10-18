from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Optional

from loguru import logger

@dataclass
class MetadataProperty:
    name: str
    dtype: Literal["STRING", "ENUM"] 
    multiple: bool  # True if can take multiple values
    nullable: bool  # True if can be None
    enum_options: set[str] | None = None  # For ENUM types

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
        self.property_map = {prop.name: prop for prop in self.get_properties()}
        self._internal_keys = set(["file_id", "dataset_id"])
        for key in self.property_map.keys():
            if key in self._internal_keys:
                raise RuntimeError(f"Property {key} is a Mixtera-internal key. It should not be in properties.")

    @classmethod
    @abstractmethod
    def get_properties(cls) -> list[MetadataProperty]:
        """
        Return the list of MetadataProperty objects defining the schema.
        """
        pass

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
        metadata: dict[str, Any] = {"sample_id": sample_id}

        for key, prop in self.property_map.items():
            value = kwargs.get(key, None)

            if value is None:
                if not prop.nullable:
                    raise RuntimeError(f"Property '{key}' is not nullable, but no value was provided.")
                else:
                    metadata[key] = None
                    continue

            if prop.multiple:
                if not isinstance(value, list):
                    # We might want to a warning that is only shown once here.
                    metadata[key] = [value]
                metadata[key] = value
            else:
                if isinstance(value, list):
                    raise RuntimeError(f"Property '{key}' expects a single value, but got a list.")
                metadata[key] = value

            # Validate value in case of enum
            if prop.dtype == "ENUM":
                valid_values = prop.enum_options
                if prop.multiple:
                    if not all([ele in valid_values for ele in value]):
                        raise RuntimeError(f"Property '{key}' has invalid ENUM values: {set(value) - set(valid_values)}")
                else:
                    if value not in valid_values:
                        raise RuntimeError(f"Property '{key}' has invalid ENUM value: {value}")

        self.metadata.append(metadata)