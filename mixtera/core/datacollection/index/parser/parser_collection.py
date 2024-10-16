from typing import Any, Optional

from loguru import logger
from mixtera.core.datacollection.index.parser import MetadataParser


class RedPajamaMetadataParser(MetadataParser):
    """
    Metadata parser class for the RedPajama dataset.
    """

    target_index_fields = ["language", "publication_date"]

    def parse(self, line_number: int, payload: Any, **kwargs: Optional[dict[Any, Any]]) -> None:
        if "meta" not in payload:
            return

        raw_metadata = payload["meta"]

        if not raw_metadata:
            return

        metadata = {}
        for index_field in RedPajamaMetadataParser.target_index_fields:
            if index_field not in raw_metadata:
                continue
            value = raw_metadata[index_field]
            if index_field == "language":
                metadata["language"] = []
                for language in value:
                    metadata["language"].append(language["name"])
            else:
                # TODO(#11): Support numerical buckets, not just categorical
                metadata[index_field] = value

        self.add_metadata(sample_id=line_number, **metadata)


class SlimPajamaMetadataParser(MetadataParser):
    """
    Metadata parser class for the SlimPajama dataset.
    """

    def parse(self, line_number: int, payload: Any, **kwargs: Optional[dict[Any, Any]]) -> None:
        if "meta" not in payload:
            return

        raw_metadata = payload["meta"]

        if not raw_metadata:
            return

        self.add_metadata(sample_id=line_number, redpajama_set_name=raw_metadata["redpajama_set_name"])


class MetadataParserFactory:
    """Handles the creation of metadata parsers."""

    def __init__(self) -> None:
        # Stores the name of the parser, and its associated class
        self._registry = {"RED_PAJAMA": RedPajamaMetadataParser, "SLIM_PAJAMA": SlimPajamaMetadataParser}

    def add_parser(self, parser_name: str, parser: type[MetadataParser], overwrite: bool = False) -> bool:
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
            return True

        logger.warning(
            f"Could not register medata parser {parser_name} as "
            "it already exists with the associated class "
            f"{self._registry[parser_name]}!"
        )
        return False

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
