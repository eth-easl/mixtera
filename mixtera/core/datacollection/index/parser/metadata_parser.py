from abc import ABC, abstractmethod
from enum import IntEnum, auto
from typing import Any, Optional, Type

from mixtera.core.datacollection.index.index_collection import IndexFactory, IndexTypes, InMemoryDictionaryIndex


class MetadataParserType(IntEnum):
    """
    Enum for metadata parser types.
    """

    GENERIC_METADATA_PARSER = auto()
    RED_PAJAMA_METADATA_PARSER = auto()

    def instantiate(self) -> Type["MetadataParser"]:
        if self == MetadataParserType.RED_PAJAMA_METADATA_PARSER:
            from mixtera.core.datacollection.index.parser import (  # pylint: disable=import-outside-toplevel
                RedPajamaMetadataParser,
            )

            return RedPajamaMetadataParser

        if self == MetadataParserType.GENERIC_METADATA_PARSER:
            return MetadataParser

        raise NotImplementedError(f"Metadata parser type {self} not yet supported")


class MetadataParser(ABC):
    """
    Base class for parsing metadata. This class should be extended for parsing
    specific data collections' metadata. This class should be instantiated for
    each dataset and file, potentially allowing for parallel processing. When
    the object has completed its parsing job, the `mark_complete` method should
    be called. This compresses the index transparently.
    """

    type: MetadataParserType = MetadataParserType.GENERIC_METADATA_PARSER

    @staticmethod
    def from_type_id(type_id: int) -> Type["MetadataParser"]:
        """
        This method instantiates a metadata parser from an integer type ID (e.g., stored in a DB).

        Args:
            type_id (int): Type ID that uniquely identifies the metadata parser

        Returns:
            The class that belongs to the type_id.
        """
        try:
            metadata_parser_type = MetadataParserType(type_id)
            return metadata_parser_type.instantiate()
        except ValueError as exc:
            raise RuntimeError(f"Invalid type id {type_id}") from exc

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
        self._finalized = False
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
        assert (
            self._finalized
        ), "Retrieving index without first marking parsing as complete. Index will be transparently compressed!"
        return self._index

    def finalize(self) -> None:
        """
        Mark the completion of the metadata parsing process and convert the inner index
        to a row range-based representation.
        """
        if not self._finalized:
            self._finalized = True
            self._index = self._index.compress()

    @property
    def is_finalized(self) -> bool:
        """
        True if the parsing has been finalized, and the underlying index has
        been converted to ranges.
        """
        return self._finalized
