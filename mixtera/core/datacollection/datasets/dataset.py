from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Iterable, Optional, Type

from mixtera.core.datacollection.datasets.dataset_type import DatasetType
from mixtera.core.datacollection.index.parser import MetadataParser
from mixtera.network.connection import ServerConnection


class Dataset(ABC):
    type: DatasetType = DatasetType.GENERIC_DATASET

    @staticmethod
    def from_type_id(type_id: int) -> "Type[Dataset]":  # pylint: disable=too-many-return-statements
        """
        This method instantiates a dataset from an integer type ID (e.g., stored in a DB).

        Args:
            type_id (int): Type ID that uniquely identifies the dataset

        Returns:
            The class that belongs to the type_id.
        """
        try:
            dataset_type = DatasetType(type_id)

            if dataset_type == DatasetType.JSONL_DATASET:
                from mixtera.core.datacollection.datasets import JSONLDataset  # pylint: disable=import-outside-toplevel

                return JSONLDataset
            if dataset_type == DatasetType.WEB_DATASET:
                from mixtera.core.datacollection.datasets import WebDataset  # pylint: disable=import-outside-toplevel

                return WebDataset
            if dataset_type == DatasetType.CC12M_DATASET:
                from mixtera.core.datacollection.datasets import CC12MDataset  # pylint: disable=import-outside-toplevel

                return CC12MDataset
            if dataset_type == DatasetType.MSCOCO_DATASET:
                from mixtera.core.datacollection.datasets import (  # pylint: disable=import-outside-toplevel
                    MSCOCODataset,
                )

                return MSCOCODataset
            if dataset_type == DatasetType.LAION400M_DATASET:
                from mixtera.core.datacollection.datasets import (  # pylint: disable=import-outside-toplevel
                    LAION400MDataset,
                )

                return LAION400MDataset
            if dataset_type == DatasetType.COYO700M_DATASET:
                from mixtera.core.datacollection.datasets import (  # pylint: disable=import-outside-toplevel
                    COYO700MDataset,
                )

                return COYO700MDataset
            if dataset_type == DatasetType.DOMAINNET_DATASET:
                from mixtera.core.datacollection.datasets import (  # pylint: disable=import-outside-toplevel
                    DomainNetDataset,
                )

                return DomainNetDataset
            if dataset_type == DatasetType.PARQUET_DATASET:
                from mixtera.core.datacollection.datasets import (  # pylint: disable=import-outside-toplevel
                    ParquetDataset,
                )

                return ParquetDataset
            if dataset_type == DatasetType.GENERIC_DATASET:
                return Dataset

            raise NotImplementedError(f"Dataset type {dataset_type.name} not yet supported")
        except ValueError as exc:
            raise RuntimeError(f"Invalid type id {type_id}") from exc

    @staticmethod
    @abstractmethod
    def inform_metadata_parser(loc: Path, metadata_parser: MetadataParser) -> None:
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
    def iterate_files(loc: str) -> Iterable[str]:
        """
        Returns iterator over all files in the dataset.
        Note that this assumes the dataset to be available on th

        Args:
            loc (str): Path where the dataset is stored (can be directory or single file)

        Returns:
            Iterable over the files.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def read_ranges_from_files(
        ranges_per_file: dict[str, list[tuple[int, int]]],
        parsing_func: Callable[[str], str],
        server_connection: Optional[ServerConnection],
    ) -> Iterable[str]:
        """
        Given a list of ranges per file, iterates over the according files and yields all samples in the file.

        Args:
            ranges_per_file (dict[str, list[tuple[int, int]]]): Dict that maps file paths as keys to
                a list of ranges.
            parsing_func (Callable[[str], str]): Function applied to each "unit" per file.
                Exact meaning depends on the dataset type.
                For the JSONLDataset, this is applied per line, and can parse the actual content out of the line.
            server_connection (Optional[ServerConnection]): If not None, an open ServerConnection to the
                Mixtera server from which the file is fetched instead. If None, the file is read from the
                client directly.

        Returns:
            Iterable over the samples.
        """
        raise NotImplementedError()
