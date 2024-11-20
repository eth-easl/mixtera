from typing import Any, Optional

from loguru import logger
from mixtera.core.datacollection.index.parser import MetadataParser, MetadataProperty


class RedPajamaMetadataParser(MetadataParser):
    @classmethod
    def get_properties(cls) -> list[MetadataProperty]:
        return [
            MetadataProperty(name="language", dtype="STRING", multiple=True, nullable=True),
            MetadataProperty(name="publication_date", dtype="STRING", multiple=False, nullable=True),
        ]

    def parse(self, line_number: int, payload: Any, **kwargs: Optional[dict[Any, Any]]) -> None:
        metadata: dict[str, list] = {}
        languages = payload.get("meta", {}).get("language", [])
        if languages:
            metadata["language"] = [lang["name"] for lang in languages]
        publication_date = payload.get("meta", {}).get("publication_date")
        if publication_date:
            metadata["publication_date"] = publication_date

        self.add_metadata(sample_id=line_number, **metadata)


class SlimPajamaMetadataParser(MetadataParser):
    """
    Metadata parser class for the SlimPajama dataset.
    """

    @classmethod
    def get_properties(cls) -> list[MetadataProperty]:
        return [
            MetadataProperty(
                name="redpajama_set_name",
                dtype="ENUM",
                multiple=False,
                nullable=False,
                enum_options={
                    "RedPajamaCommonCrawl",
                    "RedPajamaC4",
                    "RedPajamaWikipedia",
                    "RedPajamaStackExchange",
                    "RedPajamaGithub",
                    "RedPajamaArXiv",
                    "RedPajamaBook",
                },
            ),
        ]

    def parse(self, line_number: int, payload: Any, **kwargs: Optional[dict[Any, Any]]) -> None:
        redpajama_set_name = payload.get("meta", {}).get("redpajama_set_name")
        if redpajama_set_name is None:
            raise RuntimeError("Property 'redpajama_set_name' is not nullable and is missing.")

        self.add_metadata(sample_id=line_number, redpajama_set_name=redpajama_set_name)


class ImagenetWebDatasetMetadataParser(MetadataParser):
    """
    Metadata parser class for the ImageNet dataset in WebDataset format.
    """

    @classmethod
    def get_properties(cls) -> list[MetadataProperty]:
        return [
            MetadataProperty(
                name="class_label",
                dtype="STRING",
                multiple=False,
                nullable=False,
            )
        ]

    def parse(self, line_number: int, payload: Any, **kwargs: Optional[dict[Any, Any]]) -> None:
        class_label = str(payload.get(".cls"))
        if class_label is None:
            raise RuntimeError("Property 'class_label' is not nullable and is missing.")

        self.add_metadata(sample_id=line_number, class_label=class_label)


class FineWebMetadataParser(MetadataParser):

    @classmethod
    def get_properties(cls) -> list[MetadataProperty]:
        return [
            MetadataProperty(name="dump", dtype="STRING", multiple=False, nullable=False),
            MetadataProperty(name="language", dtype="ENUM", enum_options=["en"], multiple=False, nullable=False),
        ]

    def parse(self, line_number: int, payload: Any, **kwargs: Optional[dict[Any, Any]]) -> None:
        language = payload.get("language")
        dump = payload.get("dump")

        metadata = {
            "dump": dump,
            "language": language,
        }

        self.add_metadata(sample_id=line_number, **metadata)


class MetadataParserFactory:
    """Handles the creation of metadata parsers."""

    def __init__(self) -> None:
        # Stores the name of the parser, and its associated class
        self._registry: dict[str, type[MetadataParser]] = {
            "RED_PAJAMA": RedPajamaMetadataParser,
            "SLIM_PAJAMA": SlimPajamaMetadataParser,
            "IMAGENET_WEB_DATASET": ImagenetWebDatasetMetadataParser,
            "FINEWEB": FineWebMetadataParser,
        }

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
