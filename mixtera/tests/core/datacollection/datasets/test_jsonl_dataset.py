import gzip
import tempfile
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

from mixtera.core.datacollection.datasets import JSONLDataset


class TestJSONLDataset(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_iterate_files_directory(self):
        directory = Path(self.temp_dir.name)
        jsonl_file_path = directory / "temp.jsonl"
        jsonl_file_path.touch()
        jsonl_file_path = directory / "temp2.jsonl"
        jsonl_file_path.touch()

        self.assertListEqual(
            sorted(list(JSONLDataset.iterate_files(str(directory)))),
            sorted([str(directory / "temp.jsonl"), str(directory / "temp2.jsonl")]),
        )

    def test_iterate_files_singlefile(self):
        directory = Path(self.temp_dir.name)
        jsonl_file_path = directory / "temp.jsonl"
        jsonl_file_path.touch()

        self.assertListEqual(list(JSONLDataset.iterate_files(jsonl_file_path)), [directory / "temp.jsonl"])

    def test_build_file_index(self):
        pass  # TODO(#8): actually write a reasonable test when it is not hardcoded anymore.

    def read_ranges_from_files_open_mock(self):
        m = mock_open(read_data='{"id": "1"}\n{"id": "2"}\n{"id": "3"}\n')
        with patch("builtins.open", m):
            ranges = [(0, 2), (2, 3)]  # Read first two lines, then the third line
            result = list(JSONLDataset._read_ranges_from_file("dummy_file.json", ranges, lambda x: x.strip(), None))
            expected = ['{"id": "1"}', '{"id": "2"}', '{"id": "3"}']
            self.assertEqual(result, expected)

    def test_read_ranges_from_files_mocked(self):
        file_contents = {
            "file://file1.json": '{"id": "1"}\n{"id": "2"}\n',
            "file://file2.json": '{"id": "3"}\n{"id": "4"}\n',
        }
        ranges_per_file = {"file://file1.json": [(0, 2)], "file://file2.json": [(0, 2)]}
        expected = ['{"id": "1"}', '{"id": "2"}', '{"id": "3"}', '{"id": "4"}']

        def side_effect(*args, **kwargs):  # pylint:disable=unused-argument
            return mock_open(read_data=file_contents[args[0]]).return_value

        with patch("mixtera.core.filesystem.local_filesystem.xopen", side_effect=side_effect):
            result = list(JSONLDataset.read_ranges_from_files(ranges_per_file, lambda x: x.strip(), None))
            self.assertEqual(result, expected)

    def test_read_ranges_from_files_e2e(self):
        # Setup: Create temporary files with content
        file1_content = '{"id": "1"}\n{"id": "2"}\n{"id": "3"}\n{"id": "4"}\n{"id": "5"}\n'
        file2_content = '{"id": "6"}\n{"id": "7"}\n{"id": "8"}\n'

        file1_path = f"{self.temp_dir.name}/temp_file1.json"
        file2_path = f"{self.temp_dir.name}/temp_file2.json"

        with open(file1_path, "w", encoding="utf-8") as f:
            f.write(file1_content)
        with open(file2_path, "w", encoding="utf-8") as f:
            f.write(file2_content)

        # Define the ranges for each file
        # For file1, we select the first two lines, skip one, then take the next two lines
        # For file2, we select only the first line
        ranges_per_file = {
            file1_path: [(0, 2), (3, 5)],
            file2_path: [(0, 1)],
        }
        expected = ['{"id": "1"}', '{"id": "2"}', '{"id": "4"}', '{"id": "5"}', '{"id": "6"}']

        # Test
        result = list(JSONLDataset.read_ranges_from_files(ranges_per_file, lambda x: x.strip(), None))
        self.assertEqual(result, expected)

    def test_read_ranges_from_files_e2e_compressed(self):
        file1_content = '{"id": "1"}\n{"id": "2"}\n{"id": "3"}\n{"id": "4"}\n{"id": "5"}\n'
        file2_content = '{"id": "6"}\n{"id": "7"}\n{"id": "8"}\n'

        file1_path = f"{self.temp_dir.name}/temp_file1.json.gz"
        file2_path = f"{self.temp_dir.name}/temp_file2.json.gz"

        # Write content to compressed files
        with gzip.open(file1_path, "wt", encoding="utf-8") as f:
            f.write(file1_content)
        with gzip.open(file2_path, "wt", encoding="utf-8") as f:
            f.write(file2_content)

        ranges_per_file = {  # As before
            file1_path: [(0, 2), (3, 5)],
            file2_path: [(0, 1)],
        }
        expected = ['{"id": "1"}', '{"id": "2"}', '{"id": "4"}', '{"id": "5"}', '{"id": "6"}']

        # Test
        result = list(JSONLDataset.read_ranges_from_files(ranges_per_file, lambda x: x.strip(), None))
        self.assertEqual(result, expected)
