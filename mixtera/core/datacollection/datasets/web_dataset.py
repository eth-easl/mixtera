import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Optional

import wids
from loguru import logger
from mixtera.core.datacollection.datasets import Dataset, DatasetType
from mixtera.core.datacollection.index.parser import MetadataParser
from mixtera.core.filesystem import FileSystem
from mixtera.network.connection import ServerConnection
from mixtera.utils.webdataset_utils import IndexedTarSamples


class WebDataset(Dataset):
    type: DatasetType = DatasetType.WEB_DATASET

    @staticmethod
    def iterate_files(loc: str) -> Iterable[str]:
        if not FileSystem.is_dir(loc):
            yield loc

        logger.info(f"Starting to iterate over samples in folder: {loc}")

        yield from FileSystem.get_all_files_with_ext(loc, "tar")

    @staticmethod
    def inform_metadata_parser(loc: Path, metadata_parser: MetadataParser) -> None:
        samples = IndexedTarSamples(str(loc), decode_images=False)

        for idx, sample in enumerate(samples):
            metadata_parser.parse(line_number=idx, payload=sample)

        samples.close()

    @staticmethod
    def read_ranges_from_files(
        ranges_per_file: dict[str, list[tuple[int, int]]],
        parsing_func: Callable[[str | dict], str],  # Will not necessarily take a string?
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
        samples = IndexedTarSamples(file)

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

        samples.close()
