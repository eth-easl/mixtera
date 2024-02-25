import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

from loguru import logger
from mixtera.core.datacollection import IndexType
from mixtera.core.datacollection.datasets import Dataset
from mixtera.core.datacollection.index import MetadataParser


class JSONLDataset(Dataset):
    type_id = 1

    @staticmethod
    def iterate_files(loc: str) -> Iterable[Path]:
        path = Path(loc)

        if not path.exists():
            raise RuntimeError(f"Path {path} does not exist.")

        is_directory = path.is_dir()

        if not is_directory:
            if not JSONLDataset._is_valid_jsonl(path):
                raise RuntimeError(
                    f"Path {path} does not belong to a directory and does not refer to a valid jsonl file."
                )

            yield path

        yield from path.glob("*.jsonl")

    @staticmethod
    def build_file_index(loc: Path, dataset_id: int, file_id: int,
                         metadata_parser: MetadataParser) -> IndexType:
        with open(loc, encoding="utf-8") as fd:
            for line_id, line in enumerate(fd):
                json_obj = json.loads(line)
                if "meta" in json_obj:
                    metadata_parser.parse(line_id, json_obj["meta"])
        return metadata_parser.get_index()

    @staticmethod
    def read_ranges_from_files(
        ranges_per_file: dict[str, list[tuple[int, int]]], parsing_func: Callable[[str], str]
    ) -> Iterable[str]:
        for file, range_list in ranges_per_file.items():
            yield from JSONLDataset.read_ranges_from_file(file, range_list, parsing_func)

    @staticmethod
    def read_ranges_from_file(
        file: str, range_list: list[tuple[int, int]], parsing_func: Callable[[str], str]
    ) -> Iterable[str]:
        with open(file, "r", encoding="utf-8") as text_file:
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
                        text_file.readline()
                    last_line_read = r_start

                # Yield the lines in the current range
                yield from (parsing_func(line) for line in itertools.islice(text_file, r_end - r_start))
                last_line_read = r_end

    @staticmethod
    def _is_valid_jsonl(path: Path) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON at line {line_number}: {e}")
                        return False
            return True
        except IOError as e:
            logger.error(f"IO error: {e}")
            return False
