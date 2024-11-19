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
            total_row_groups = parquet_file.num_row_groups

            # Collect row group offsets
            row_group_offsets = []
            current_row = 0
            for i in range(total_row_groups):
                num_rows = parquet_file.metadata.row_group(i).num_rows
                row_group_offsets.append((current_row, current_row + num_rows))
                current_row += num_rows

            # Map ranges to row groups
            # Create a mapping from row group index to list of (start_row, end_row) tuples

            row_group_ranges: dict[int, list[tuple[int, int]]] = {}
            range_idx = 0
            rg_idx = 0
            n_ranges = len(range_list)
            n_row_groups = len(row_group_offsets)

            while range_idx < n_ranges and rg_idx < n_row_groups:
                start_row, end_row = range_list[range_idx]
                rg_start, rg_end = row_group_offsets[rg_idx]

                if end_row <= rg_start:
                    # Range ends before the row group starts; move to next range
                    range_idx += 1
                elif start_row >= rg_end:
                    # Range starts after the row group ends; move to next row group
                    rg_idx += 1
                else:
                    # Overlap exists between range and row group
                    rg_overlap_start = max(start_row, rg_start)
                    rg_overlap_end = min(end_row, rg_end)

                    # Compute the relative start and end within the row group
                    relative_start = rg_overlap_start - rg_start
                    relative_end = rg_overlap_end - rg_start

                    if rg_idx not in row_group_ranges:
                        row_group_ranges[rg_idx] = []
                    row_group_ranges[rg_idx].append((relative_start, relative_end))

                    # Decide which pointer to advance
                    if end_row <= rg_end:
                        # Range ends before or at the end of the row group; move to next range
                        range_idx += 1
                    else:
                        # Row group ends before the range; move to next row group
                        rg_idx += 1

            # Read and process relevant row groups
            for rg_index, ranges in row_group_ranges.items():
                table = parquet_file.read_row_group(rg_index)
                for start, end in ranges:
                    length = end - start
                    sliced_table = table.slice(start, length)

                    for batch in sliced_table.to_batches(max_chunksize=32000):
                        struct_array = batch.to_struct_array()
                        for record in struct_array:
                            yield parsing_func(record.as_py())

    @staticmethod
    def _is_valid_parquet(path: str) -> bool:
        try:
            with open(path, "rb") as file:
                pq.ParquetFile(file)
            return True
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Invalid Parquet file {path}: {e}")
            return False
