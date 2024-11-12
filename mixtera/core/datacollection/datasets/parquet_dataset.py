from pathlib import Path
from typing import Callable, Iterable, Optional

import pyarrow.parquet as pq
from loguru import logger
from mixtera.core.datacollection.datasets import Dataset, DatasetType
from mixtera.core.datacollection.index.parser import MetadataParser
from mixtera.core.filesystem import FileSystem
from mixtera.network.connection import ServerConnection


class ParquetDataset(Dataset):
    type: DatasetType = DatasetType.PARQUET_DATASET

    @staticmethod
    def iterate_files(loc: str) -> Iterable[str]:
        if not FileSystem.is_dir(loc):
            if not ParquetDataset._is_valid_parquet(loc):
                raise RuntimeError(
                    f"Path {loc} does not belong to a directory and does not refer to a valid parquet file."
                )
            yield loc
        else:
            yield from FileSystem.get_all_files_with_ext(loc, "parquet")

    @staticmethod
    def inform_metadata_parser(loc: Path, metadata_parser: MetadataParser) -> None:
        with open(loc, "rb") as f:
            parquet_file = pq.ParquetFile(f)
            row_id = 0
            for batch in parquet_file.iter_batches():
                records = batch.to_pylist()
                for record in records:
                    metadata_parser.parse(row_id, record)
                    row_id += 1

    @staticmethod
    def read_ranges_from_files(
        ranges_per_file: dict[str, list[tuple[int, int]]],
        parsing_func: Callable[[dict], str],
        server_connection: Optional[ServerConnection],
    ) -> Iterable[str]:
        for file, range_list in ranges_per_file.items():
            yield from ParquetDataset._read_ranges_from_file(file, range_list, parsing_func, server_connection)

    @staticmethod
    def _read_ranges_from_file(
        file: str,
        range_list: list[tuple[int, int]],
        parsing_func: Callable[[dict], str],
        server_connection: Optional[ServerConnection],
    ) -> Iterable[str]:
        del server_connection  # TODO(#137): We need a open interface with a regular file object to use that.
        with open(file, "rb") as f:
            parquet_file = pq.ParquetFile(f)

            range_iter = iter(range_list)
            if (current_range := next(range_iter, None)) is None:
                return  # No ranges to process

            start_row, end_row = current_range  # Ranges are [start_row, end_row)
            row_id = 0

            for batch in parquet_file.iter_batches():
                batch_size = len(batch)
                batch_start_row = row_id
                batch_end_row = row_id + batch_size

                if batch_end_row <= start_row:
                    # Skip the entire batch
                    row_id += batch_size
                    continue

                if batch_start_row >= end_row:
                    # Move to the next range
                    if (current_range := next(range_iter, None)) is None:
                        return
                    start_row, end_row = current_range
                    if batch_end_row <= start_row:
                        # Skip the batch if it doesn't overlap with the new range
                        row_id += batch_size
                        continue

                batch_records = batch.to_pylist()
                for i, record in enumerate(batch_records):
                    current_row_id = batch_start_row + i

                    if current_row_id >= end_row:
                        # We've reached the end of the current range
                        if (current_range := next(range_iter, None)) is None:
                            return
                        start_row, end_row = current_range

                        if current_row_id < start_row:
                            # Skip records until we reach the start of the next range
                            continue

                    if start_row <= current_row_id < end_row:
                        yield parsing_func(record)
                row_id += batch_size

    @staticmethod
    def _is_valid_parquet(path: str) -> bool:
        try:
            with open(path, "rb") as file:
                pq.ParquetFile(file)
            return True
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Invalid Parquet file {path}: {e}")
            return False
