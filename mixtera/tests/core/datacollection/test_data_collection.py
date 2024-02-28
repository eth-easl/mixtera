import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from mixtera.core.datacollection import MixteraDataCollection


class TestMixteraDataCollection(unittest.TestCase):

    @patch("mixtera.core.datacollection.local.LocalDataCollection")
    def test_from_directory_with_existing_dir(self, mock_local_data_collection):
        mock_local_data_collection.return_value = MagicMock()

        dir_path = Path(".")
        result = MixteraDataCollection.from_directory(dir_path)
        mock_local_data_collection.assert_called_once_with(dir_path)
        self.assertIsInstance(result, MagicMock)

    def test_from_directory_with_non_existing_dir(self):
        dir_path = Path("/non/existing/directory")
        with self.assertRaises(RuntimeError):
            MixteraDataCollection.from_directory(dir_path)

    def test_from_remote(self):
        pass
        # raise NotImplementedError("This test needs to be written")
