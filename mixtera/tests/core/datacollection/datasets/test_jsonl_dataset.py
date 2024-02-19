import tempfile
import unittest
from pathlib import Path

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
            list(JSONLDataset.iterate_files(str(directory))), [directory / "temp.jsonl", directory / "temp2.jsonl"]
        )

    def test_iterate_files_singlefile(self):
        directory = Path(self.temp_dir.name)
        jsonl_file_path = directory / "temp.jsonl"
        jsonl_file_path.touch()

        self.assertListEqual(list(JSONLDataset.iterate_files(jsonl_file_path)), [directory / "temp.jsonl"])

    def test_build_file_index(self):
        pass  # TODO(#8): actually write a reasonable test when it is not hardcoded anymore.
