import itertools
import json
from pathlib import Path
from typing import Callable, Iterable, Optional

import ijson
from loguru import logger
from mixtera.core.datacollection.datasets import Dataset, DatasetType
from mixtera.core.datacollection.index.parser import MetadataParser
from mixtera.core.filesystem import FileSystem
from mixtera.network.connection import ServerConnection


class LLaVADataset(Dataset):
    type: DatasetType = DatasetType.LLAVA_DATASET
    dataset_name: str = "LLaVA"

    @staticmethod
    def iterate_files(loc: str) -> Iterable[str]:
        if not FileSystem.is_dir(loc):
            if not LLaVADataset._is_valid_llava_json(loc):
                raise RuntimeError(
                    f"Path {loc} does not belong to a directory and does not refer to a valid llava json file."
                )

            yield loc

        yield from FileSystem.get_all_files_with_exts(loc, ["json"])

    @staticmethod
    def inform_metadata_parser(loc: Path, metadata_parser: MetadataParser) -> None:
        with open(loc) as file:
            dataset = json.load(file)
            for idx, sample in enumerate(dataset):
                metadata_parser.parse(line_number=idx, payload=sample, dataset_name=LLaVADataset.dataset_name)

    @staticmethod
    def read_ranges_from_files(
        ranges_per_file: dict[str, list[tuple[int, int]]],
        parsing_func: Callable[[str], str],
        server_connection: Optional[ServerConnection],
    ) -> Iterable[str]:
        for file, range_list in ranges_per_file.items():
            yield from LLaVADataset._read_ranges_from_file(file, range_list, parsing_func, server_connection)

    @staticmethod
    def _read_ranges_from_file(  # pylint: disable=contextmanager-generator-missing-cleanup
        file: str,
        range_list: list[tuple[int, int]],
        parsing_func: Callable[[str], str],
        server_connection: Optional[ServerConnection],
    ) -> Iterable[str]:
        with open(file) as json_file:
            samples = ijson.items(json_file, "item")

            last_line_read = 0
            last_r_start = -1
            for r_start, r_end in range_list:
                if r_start < last_r_start:
                    raise RuntimeError(f"Ranges not sorted by start ({last_r_start} vs {r_start})")

                if last_line_read > r_start:
                    raise RuntimeError(f"Overlapping ranges: start at {r_start} but previous ended at {last_line_read}")

                last_r_start = r_start

                # Skip lines to reach the start of the new range if necessary
                if r_start > last_line_read:
                    for _ in range(r_start - last_line_read):
                        next(samples)
                    last_line_read = r_start

                # Yield the lines in the current range
                yield from (parsing_func(line) for line in itertools.islice(samples, r_end - r_start))
                last_line_read = r_end

    @staticmethod
    def _is_valid_llava_json(path: str) -> bool:
        try:
            with open(path) as file:
                samples = ijson.items(file, "samples")
                for sample in samples:
                    if "id" not in sample or "image" not in sample or "conversations" not in sample:
                        return False
            return True
        except IOError as e:
            logger.error(f"IO error: {e}")
            return False
