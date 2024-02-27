from enum import Enum
from typing import Any, Optional

from loguru import logger
from mixtera.core.datacollection import IndexType
from mixtera.core.datacollection.index import MetadataParser
from mixtera.utils import ranges


class RedPajamaMetadataParser(MetadataParser):
    """
    Metadata parser class for the RedPajama dataset.
    """

    target_index_fields = ["language", "publication_date"]

    def parse(self, line_number: int, metadata: Any, **kwargs: Optional[dict[Any, Any]]) -> None:
        for index_field in RedPajamaMetadataParser.target_index_fields:
            if index_field not in metadata:
                continue
            value = metadata[index_field]
            if index_field == "language":
                for language in value:
                    self._index[index_field][language["name"]][self.dataset_id][self.file_id].append(line_number)
            else:
                # TODO(#11): Support numerical buckets, not just categorical
                self._index[index_field][value][self.dataset_id][self.file_id].append(line_number)

    def _compress_index(self) -> None:
        """
        Compresses the internal index, reducing contiguous line ranges to spans.
        E.g. [1,2,3,5,6] --> [(1,4), (5,7)]. All modifications are done in place
        on the index. Note that the lower bound of each range is inclusive, but
        the upper bound is exclusive.
        """
        for _, buckets in self._index.items():
            for __, bucket_vals in buckets.items():
                bucket_vals[self.dataset_id][self.file_id] = ranges(bucket_vals[self.dataset_id][self.file_id])

    def get_index(self) -> IndexType:
        self._compress_index()
        return self._index


class MetadataParserRegistry(Enum):
    """Each metadata parser is registered with this enum."""

    RED_PAJAMA = RedPajamaMetadataParser


class MetadataParserFactory:
    """Handles the creation of metadata parsers."""

    def __init__(self):
        # Stores the name of the parser, and its associated class
        self._registry = {"RED_PAJAMA": RedPajamaMetadataParser}

    def add_parser(self, parser_name, parser: type[MetadataParser], overwrite=False) -> None:
        """
        Add a new metadata parser to the factory. If parser already exists
        at name `parser_name`, but `overwrite` is `True`, then overwrite it.

        Args:
            parser_name: the name of the metadata parser
            parser: A subclass of MetadataParser
            overwrite: whether to overwrite an existing metadata parser
        """
        if parser_name not in self._registry or overwrite:
            self._registry[parser_name] = parser
            logger.info(f"Registered medata parser {parser_name} with the " f"associated class {parser}")
        else:
            logger.warning(
                f"Could not register medata parser {parser_name} as "
                "it already exists with the associated class "
                f"{self._registry[parser_name]}!"
            )

    def remove_parser(self, parser_name: str) -> None:
        """
        Remove a metadata parser.

        Args:
            parser_name: The name of the metadata parser to be removed
        """
        if parser_name in self._registry:
            del self._registry[parser_name]
            logger.info(f"Removed medata parser {parser_name}")
        else:
            logger.info(f"Tried to remove medata parser {parser_name} but it " "does not exist in the registry!")

    def create_metadata_parser(self, parser_name: str, dataset_id: int, file_id: int) -> MetadataParser:
        """
        Factory method that creates a `parser_name` metadata parser. If no
        parser is registered under `parser_name`, this method throws a
        `ModuleNotFoundError`.

        Args:
            parser_name: name of the metadata parser to be instantiated
            dataset_id: the id of the source dataset
            file_id: id of the parsed file

        Returns:
            A `MetadataParser` typed object or raises an error if `parser_name`
            does not exist
        """
        if parser_name in self._registry:
            return self._registry[parser_name](dataset_id, file_id)
        error_msg = f"Could not create {parser_name} metadata parser as it " "does not exist in the registry!"
        logger.error(error_msg)
        raise ModuleNotFoundError(error_msg)
