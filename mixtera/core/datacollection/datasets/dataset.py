from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Iterable, Type

from mixtera.core.datacollection.index.parser import MetadataParser


class Dataset(ABC):
    type_id = 0

    @staticmethod
    def from_type_id(type_id: int) -> "Type[Dataset]":
        """
        This method instantiates a dataset from an integer type ID (e.g., stored in a DB).

        Args:
            type_id (int): Type ID that uniquely identifies the dataset

        Returns:
            The class that belongs to the type_id.
        """
        if type_id < 1:
            raise RuntimeError(f"Invalid type id {type_id}")

        from mixtera.core.datacollection.datasets.jsonl_dataset import (  # pylint: disable=import-outside-toplevel
            JSONLDataset,
        )

        if type_id == JSONLDataset.type_id:
            return JSONLDataset

        raise NotImplementedError(f"type_id {type_id} not yet supported")

    @staticmethod
    @abstractmethod
    def build_file_index(loc: Path, metadata_parser: MetadataParser) -> None:
        """
        Build up the file index for the file stored at loc.

        Args:
            loc (Path): Path to the file we are building the index for
            metadata_parser (MetadataParser): Parser class responsible with extracting the metadata.
                This object is stateful.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def iterate_files(loc: str) -> Iterable[Path]:
        """
        Returns iterator over all files in the dataset.

        Args:
            loc (str): Path where the dataset is stored (can be directory or single file)

        Returns:
            Iterable over Paths.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def read_ranges_from_files(
        ranges_per_file: dict[str, list[tuple[int, int]]], parsing_func: Callable[[str], str]
    ) -> Iterable[str]:
        """
        Given a list of ranges per file, iterates over the according files and yields all samples in the file.

        Args:
            ranges_per_file (dict[str, list[tuple[int, int]]]): Dict that maps file paths as keys to
                a list of ranges.
            parsing_func (Callable[[str], str]): Function applied to each "unit" per file.
                Exact meaning depends on the dataset type.
                For the JSONLDataset, this is applied per line, and can parse the actual content out of the line.

        Returns:
            Iterable over the samples.
        """
        raise NotImplementedError()
