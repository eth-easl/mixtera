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
            row_group_ranges = {}
            for start_row, end_row in range_list:
                for rg_index, (rg_start, rg_end) in enumerate(row_group_offsets):
                    # Check if the range overlaps with the row group
                    if end_row <= rg_start:
                        # Range ends before this row group starts
                        break  # Since row groups are in order, no need to check further
                    if start_row >= rg_end:
                        # Range starts after this row group ends
                        continue
                    # Overlap exists
                    rg_start_row = max(start_row, rg_start)
                    rg_end_row = min(end_row, rg_end)
                    if rg_index not in row_group_ranges:
                        row_group_ranges[rg_index] = []
                    row_group_ranges[rg_index].append((rg_start_row - rg_start, rg_end_row - rg_start))

            # Read and process relevant row groups
            for rg_index, ranges in row_group_ranges.items():
                # Read only the necessary columns if possible
                table = parquet_file.read_row_group(rg_index)
                for start, end in ranges:
                    length = end - start
                    sliced_table = table.slice(start, length)

                    # Use iter_batches() to limit memory usage
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
