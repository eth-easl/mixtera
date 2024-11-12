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

    def create_sample_parquet_file(self, file_path, num_rows=10, start_id=0, schema=None):
        ids = [start_id + i for i in range(num_rows)]
        data = {"id": ids, "value": [f"value_{i}" for i in ids]}

        if schema:
            for col in schema:
                if col not in data:
                    data[col] = [f"{col}_{i}" for i in ids]

        table = pa.Table.from_pydict(data)
        pq.write_table(table, file_path)

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
        """
        Test reading ranges that result in no data (e.g., start row equals end row).
        """
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
        """
        Test the inform_metadata_parser function.
        """
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
