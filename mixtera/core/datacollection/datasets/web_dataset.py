from pathlib import Path
from typing import Callable, ClassVar, Iterable, Optional

from loguru import logger
from mixtera.core.datacollection.datasets import Dataset, DatasetType
from mixtera.core.datacollection.index.parser import MetadataParser
from mixtera.core.filesystem import FileSystem
from mixtera.network.connection import ServerConnection
from mixtera.utils.webdataset_utils import IndexedTarSamples


class WebDataset(Dataset):
    type: DatasetType = DatasetType.WEB_DATASET
    dataset_name: ClassVar[str] = "WebDataset"

    @staticmethod
    def iterate_files(loc: str) -> Iterable[str]:
        if not FileSystem.is_dir(loc):
            yield loc

        logger.info(f"Starting to iterate over samples in folder: {loc}")

        yield from FileSystem.get_all_files_with_ext(loc, "tar")

    @staticmethod
    def inform_metadata_parser(loc: Path, metadata_parser: MetadataParser) -> None:
        """Parse metadata from a WebDataset tar file."""
        cls = WebDataset  # Use the current class (works for subclasses too)
        dataset_name = getattr(cls, "dataset_name", cls.__name__)

        samples = IndexedTarSamples(str(loc))

        logger.info(f"Starting to iterate over samples ({cls.__name__}) in folder: {loc}")
        for idx, sample in enumerate(samples):
            metadata_parser.parse(
                line_number=idx,
                payload=sample,
                dataset_name=dataset_name if dataset_name != "WebDataset" else None,
            )

        samples.close()

    @staticmethod
    def read_ranges_from_files(
        ranges_per_file: dict[str, list[tuple[int, int]]],
        # Will not necessarily take a string?
        parsing_func: Callable[[str | dict], str],
        server_connection: Optional[ServerConnection],
    ) -> Iterable[str | dict]:
        for file, range_list in ranges_per_file.items():
            yield from WebDataset._read_ranges_from_file(file, range_list, parsing_func, server_connection)

    @staticmethod
    def _read_ranges_from_file(  # pylint: disable=contextmanager-generator-missing-cleanup
        file: str,
        range_list: list[tuple[int, int]],
        parsing_func: Callable[[dict], str],
        server_connection: Optional[ServerConnection],  # pylint: disable=unused-argument
    ) -> Iterable[str]:
        with IndexedTarSamples(file) as samples:
            last_line_read = 0
            last_r_start = -1
            for r_start, r_end in range_list:
                if r_start < last_r_start:
                    raise RuntimeError(f"Ranges not sorted by start ({last_r_start} vs {r_start})")

                if last_line_read > r_start:
                    raise RuntimeError(f"Overlapping ranges: start at {r_start} but previous ended at {last_line_read}")

                last_r_start = r_start

                yield from (parsing_func(samples[line]) for line in range(r_start, r_end))

                last_line_read = r_end


class CC12MDataset(WebDataset):
    type: DatasetType = DatasetType.CC12M_DATASET
    dataset_name: ClassVar[str] = "CC12M"


class MSCOCODataset(WebDataset):
    type: DatasetType = DatasetType.MSCOCO_DATASET
    dataset_name: ClassVar[str] = "MSCOCO"


class LAION400MDataset(WebDataset):
    type: DatasetType = DatasetType.LAION400M_DATASET
    dataset_name: ClassVar[str] = "LAION400M"


class COYO700MDataset(WebDataset):
    type: DatasetType = DatasetType.COYO700M_DATASET
    dataset_name: ClassVar[str] = "COYO700M"


class DomainNetDataset(WebDataset):
    type: DatasetType = DatasetType.DOMAINNET_DATASET
    dataset_name: ClassVar[str] = "DomainNet"

    @staticmethod
    def inform_metadata_parser(loc: Path, metadata_parser: MetadataParser) -> None:
        dataset_name = DomainNetDataset.dataset_name

        samples = IndexedTarSamples(str(loc))

        logger.info(f"Starting to iterate over samples (DomainNet) in folder: {loc}")
        for idx, sample in enumerate(samples):
            class_name = sample["cls"]
            domain = sample["domain"]

            metadata_parser.parse(
                line_number=idx,
                payload=sample,
                dataset_name=dataset_name,
                class_name=class_name,
                domain=domain,
            )

        samples.close()
