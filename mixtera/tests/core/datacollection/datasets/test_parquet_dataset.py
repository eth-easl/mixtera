import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from mixtera.core.datacollection.datasets import ParquetDataset


class TestParquetDataset(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.directory = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_sample_parquet_file(self, file_path, num_rows=10, start_id=0, row_group_size=None, schema=None):
        ids = [start_id + i for i in range(num_rows)]
        data = {"id": ids, "value": [f"value_{i}" for i in ids]}

        if schema:
            for col in schema:
                if col not in data:
                    data[col] = [f"{col}_{i}" for i in ids]

        table = pa.Table.from_pydict(data)

        # Write the table with specified row group size
        write_table_kwargs = {}
        if row_group_size:
            write_table_kwargs["row_group_size"] = row_group_size
        pq.write_table(table, file_path, **write_table_kwargs)

    def test_iterate_files_directory(self):
        parquet_file1 = self.directory / "file1.parquet"
        parquet_file2 = self.directory / "file2.parquet"
        self.create_sample_parquet_file(parquet_file1)
        self.create_sample_parquet_file(parquet_file2)

        expected_files = [str(parquet_file1), str(parquet_file2)]
        iterated_files = sorted(list(ParquetDataset.iterate_files(str(self.directory))))
        self.assertListEqual(iterated_files, sorted(expected_files))

    def test_iterate_files_singlefile(self):
        parquet_file = self.directory / "file.parquet"
        self.create_sample_parquet_file(parquet_file)

        iterated_files = list(ParquetDataset.iterate_files(str(parquet_file)))
        self.assertListEqual(iterated_files, [str(parquet_file)])

    def test_read_ranges_from_files_e2e(self):
        # Create sample Parquet files
        file1_path = self.directory / "file1.parquet"
        file2_path = self.directory / "file2.parquet"
        self.create_sample_parquet_file(file1_path, num_rows=5, start_id=1)  # ids 1-5
        self.create_sample_parquet_file(file2_path, num_rows=5, start_id=6)  # ids 6-10

        # Define the ranges for each file
        # For file1, read rows 0-2 (ids 1-3)
        # For file2, read rows 3-5 (ids 9-10)
        ranges_per_file = {
            str(file1_path): [(0, 2)],  # Rows 0 and 1 (ids 1 and 2)
            str(file2_path): [(3, 5)],  # Rows 3 and 4 (ids 9 and 10)
        }

        expected_records = [
            {"id": 1, "value": "value_1"},
            {"id": 2, "value": "value_2"},
            {"id": 9, "value": "value_9"},
            {"id": 10, "value": "value_10"},
        ]

        def parsing_func(record):
            return record

        results = list(ParquetDataset.read_ranges_from_files(ranges_per_file, parsing_func, None))
        self.assertEqual(results, expected_records)

    def test_read_empty_parquet_file(self):
        empty_file_path = self.directory / "empty.parquet"
        table = pa.Table.from_pydict({"id": [], "value": []})
        pq.write_table(table, empty_file_path)

        ranges_per_file = {
            str(empty_file_path): [(0, 1)],
        }

        def parsing_func(record):
            return record

        results = list(ParquetDataset.read_ranges_from_files(ranges_per_file, parsing_func, None))
        self.assertEqual(results, [])

    def test_read_large_parquet_file(self):
        # Create a large Parquet file with 10000 rows
        large_file_path = self.directory / "large.parquet"
        self.create_sample_parquet_file(large_file_path, num_rows=10000, start_id=1)

        # Define ranges to read from the large file
        ranges_per_file = {
            str(large_file_path): [(500, 505), (9995, 10000)],
        }

        expected_records = []
        for i in range(500, 505):
            expected_records.append({"id": i + 1, "value": f"value_{i + 1}"})
        for i in range(9995, 10000):
            expected_records.append({"id": i + 1, "value": f"value_{i + 1}"})

        def parsing_func(record):
            return record

        results = list(ParquetDataset.read_ranges_from_files(ranges_per_file, parsing_func, None))
        self.assertEqual(results, expected_records)

    def test_read_ranges_with_no_data(self):
        # Create a sample Parquet file
        file_path = self.directory / "file.parquet"
        self.create_sample_parquet_file(file_path, num_rows=5, start_id=1)

        # Define a range with no data
        ranges_per_file = {
            str(file_path): [(2, 2)],  # Start and end are the same
        }

        def parsing_func(record):
            return record

        results = list(ParquetDataset.read_ranges_from_files(ranges_per_file, parsing_func, None))
        self.assertEqual(results, [])

    def test_inform_metadata_parser(self):
        file_path = self.directory / "file.parquet"
        self.create_sample_parquet_file(file_path, num_rows=5, start_id=1)

        class MockMetadataParser:
            def __init__(self):
                self.records = []

            def parse(self, line_id, record):
                self.records.append((line_id, record))

        metadata_parser = MockMetadataParser()
        ParquetDataset.inform_metadata_parser(file_path, metadata_parser)

        expected_records = [
            (0, {"id": 1, "value": "value_1"}),
            (1, {"id": 2, "value": "value_2"}),
            (2, {"id": 3, "value": "value_3"}),
            (3, {"id": 4, "value": "value_4"}),
            (4, {"id": 5, "value": "value_5"}),
        ]

        self.assertEqual(metadata_parser.records, expected_records)

    def test_read_full_overlap_with_row_groups(self):
        file_path = self.directory / "row_group_overlap.parquet"
        # Create a parquet file with 10 rows and row groups of size 5
        self.create_sample_parquet_file(file_path, num_rows=10, start_id=1, row_group_size=5)

        # Define ranges that align exactly with row group boundaries
        ranges_per_file = {
            str(file_path): [(0, 5), (5, 10)],  # Valid, increasing, non-overlapping
        }

        expected_records = []
        for i in range(1, 11):
            expected_records.append({"id": i, "value": f"value_{i}"})

        def parsing_func(record):
            return record

        results = list(ParquetDataset.read_ranges_from_files(ranges_per_file, parsing_func, None))
        self.assertEqual(results, expected_records)

    def test_read_partial_overlap_with_row_groups(self):
        file_path = self.directory / "partial_row_group_overlap.parquet"
        # Create a parquet file with 15 rows and row groups of size 5
        self.create_sample_parquet_file(file_path, num_rows=15, start_id=1, row_group_size=5)

        ranges_per_file = {
            str(file_path): [(3, 8)],  # Overlaps with the first and second row groups
        }

        # Expected IDs are from 4 to 8 inclusive
        expected_records = []
        for i in range(4, 9):
            expected_records.append({"id": i, "value": f"value_{i}"})

        def parsing_func(record):
            return record

        results = list(ParquetDataset.read_ranges_from_files(ranges_per_file, parsing_func, None))
        self.assertEqual(results, expected_records)

    def test_read_ranges_at_data_bounds(self):
        file_path = self.directory / "data_bounds.parquet"
        self.create_sample_parquet_file(file_path, num_rows=10, start_id=1)

        # Define ranges at the very start and end of the data
        ranges_per_file = {
            str(file_path): [(0, 1), (9, 10)],  # Rows 0 and 9
        }

        expected_records = [
            {"id": 1, "value": "value_1"},  # Row 0
            {"id": 10, "value": "value_10"},  # Row 9
        ]

        def parsing_func(record):
            return record

        results = list(ParquetDataset.read_ranges_from_files(ranges_per_file, parsing_func, None))
        self.assertEqual(results, expected_records)

    def test_read_adjacent_ranges(self):
        file_path = self.directory / "adjacent_ranges.parquet"
        self.create_sample_parquet_file(file_path, num_rows=10, start_id=1, row_group_size=5)

        # Define adjacent ranges that touch but do not overlap
        ranges_per_file = {
            str(file_path): [(0, 5), (5, 10)],
        }

        expected_records = []
        for i in range(1, 11):
            expected_records.append({"id": i, "value": f"value_{i}"})

        def parsing_func(record):
            return record

        # The expected records are the same as the full dataset
        results = list(ParquetDataset.read_ranges_from_files(ranges_per_file, parsing_func, None))
        self.assertEqual(results, expected_records)

    def test_read_single_row_ranges(self):
        file_path = self.directory / "single_row_ranges.parquet"
        self.create_sample_parquet_file(file_path, num_rows=10, start_id=1, row_group_size=1000)

        # Define valid, increasing, non-overlapping single-row ranges
        ranges_per_file = {
            str(file_path): [(2, 3), (5, 6), (8, 9)],  # Rows 2, 5, and 8
        }

        expected_records = [
            {"id": 3, "value": "value_3"},
            {"id": 6, "value": "value_6"},
            {"id": 9, "value": "value_9"},
        ]

        def parsing_func(record):
            return record

        results = list(ParquetDataset.read_ranges_from_files(ranges_per_file, parsing_func, None))
        self.assertEqual(results, expected_records)

    def test_read_range_spanning_all_row_groups(self):
        file_path = self.directory / "all_row_groups.parquet"
        # Create a parquet file with 20 rows and row groups of size 5
        self.create_sample_parquet_file(file_path, num_rows=20, start_id=1, row_group_size=5)

        # Define a range that spans all row groups
        ranges_per_file = {
            str(file_path): [(0, 20)],
        }

        expected_records = []
        for i in range(1, 21):
            expected_records.append({"id": i, "value": f"value_{i}"})

        def parsing_func(record):
            return record

        # The expected records are the full dataset
        results = list(ParquetDataset.read_ranges_from_files(ranges_per_file, parsing_func, None))
        self.assertEqual(results, expected_records)

    def test_read_range_at_row_group_boundary(self):
        file_path = self.directory / "boundary_ranges.parquet"
        # Create a parquet file with 15 rows and row groups of size 5
        self.create_sample_parquet_file(file_path, num_rows=15, start_id=1, row_group_size=5)

        # Define a range that starts at a row group boundary and ends within the next row group
        ranges_per_file = {
            str(file_path): [(5, 8)],  # Starts at index 5 (row group boundary)
        }

        expected_records = []
        for i in range(6, 9):
            expected_records.append({"id": i, "value": f"value_{i}"})

        def parsing_func(record):
            return record

        # Expected to read records with IDs 6, 7, and 8 (because start_id=1)
        results = list(ParquetDataset.read_ranges_from_files(ranges_per_file, parsing_func, None))
        self.assertEqual(results, expected_records)
