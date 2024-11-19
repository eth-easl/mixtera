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
        del server_connection  # TODO: Implement server connection handling if needed
        with open(file, "rb") as f:
            parquet_file = pq.ParquetFile(f)

            range_iter = iter(range_list)
            current_range = next(range_iter, None)
            if current_range is None:
                return  # No ranges to process

            start_row, end_row = current_range  # Ranges are [start_row, end_row)
            row_id = 0

            for batch in parquet_file.iter_batches():
                batch_size = len(batch)
                batch_start_row = row_id
                batch_end_row = row_id + batch_size

                while True:
                    # Skip batches that end before the start of the current range
                    if batch_end_row <= start_row:
                        break  # Proceed to next batch

                    # If the batch starts after the current range ends, move to the next range
                    if batch_start_row >= end_row:
                        # Move to the next range
                        current_range = next(range_iter, None)
                        if current_range is None:
                            return  # No more ranges
                        start_row, end_row = current_range
                        continue  # Re-enter the loop with the new range

                    # Calculate overlap between batch and current range
                    overlap_start_row = max(batch_start_row, start_row)
                    overlap_end_row = min(batch_end_row, end_row)

                    if overlap_end_row <= overlap_start_row:
                        # No overlap, move to the next range
                        current_range = next(range_iter, None)
                        if current_range is None:
                            return
                        start_row, end_row = current_range
                        continue  # Re-enter the loop with the new range

                    # Slice the batch to only include overlapping rows
                    overlap_start_in_batch = overlap_start_row - batch_start_row
                    overlap_length = overlap_end_row - overlap_start_row
                    overlap_batch = batch.slice(overlap_start_in_batch, overlap_length)

                    # Convert the overlap_batch to a StructArray
                    struct_array = overlap_batch.to_struct_array()

                    # Process the overlapping batch
                    for record in struct_array:
                        yield parsing_func(record.as_py())

                    # Check if we've reached the end of the current range
                    if overlap_end_row == end_row:
                        # Move to the next range
                        current_range = next(range_iter, None)
                        if current_range is None:
                            return  # No more ranges
                        start_row, end_row = current_range
                    else:
                        # Proceed to the next batch
                        break

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
