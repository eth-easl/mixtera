import itertools
import json
from pathlib import Path
from typing import Callable, Iterable, Optional, Type, TYPE_CHECKING

from loguru import logger
from mixtera.core.datacollection import IndexType
from mixtera.core.datacollection.datasets import Dataset
from mixtera.core.datacollection.index import MetadataParser
from mixtera.core.filesystem import AbstractFilesystem

if TYPE_CHECKING:
    from mixtera.network.connection import ServerConnection


class JSONLDataset(Dataset):
    type_id = 1

    @staticmethod
    def iterate_files(loc: str, filesys_t: Type[AbstractFilesystem]) -> Iterable[str]:
        if not filesys_t.is_dir(loc):
            if not JSONLDataset._is_valid_jsonl(loc, filesys_t):
                raise RuntimeError(
                    f"Path {loc} does not belong to a directory and does not refer to a valid jsonl file."
                )

            yield loc

        yield from filesys_t.get_all_files_with_ext(loc, "jsonl")

    @staticmethod
    def build_file_index(loc: Path, filesys_t: Type[AbstractFilesystem], metadata_parser: MetadataParser) -> IndexType:
        with filesys_t.open_file(loc) as fd:
            for line_id, line in enumerate(fd):
                metadata_parser.parse(line_id, json.loads(line))
        return metadata_parser.get_index()

    @staticmethod
    def read_ranges_from_files(
        ranges_per_file: dict[str, list[tuple[int, int]]],
        filesys_t: Type[AbstractFilesystem],
        parsing_func: Callable[[str], str],
        server_connection: Optional["ServerConnection"],
    ) -> Iterable[str]:
        for file, range_list in ranges_per_file.items():
            yield from JSONLDataset._read_ranges_from_file(file, filesys_t, range_list, parsing_func, server_connection)

    @staticmethod
    def _read_ranges_from_file(
        file: str,
        filesys_t: Type[AbstractFilesystem],
        range_list: list[tuple[int, int]],
        parsing_func: Callable[[str], str],
        server_connection: Optional["ServerConnection"],
    ) -> Iterable[str]:
        with filesys_t.open_file(file, server_connection=server_connection) as text_file:
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
                        next(text_file, None)
                    last_line_read = r_start

                # Yield the lines in the current range
                yield from (parsing_func(line) for line in itertools.islice(text_file, r_end - r_start))
                last_line_read = r_end

    @staticmethod
    def _is_valid_jsonl(path: str, filesys_t: Type[AbstractFilesystem]) -> bool:
        try:
            with filesys_t.open_file(path) as file:
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
