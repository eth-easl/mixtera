import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

from mixtera.core.filesystem import LocalFileSystem


class TestLocalFileSystem(unittest.TestCase):
    def test_get_file_iterable(self):
        file_path = "testfile.txt"
        mock_file_data = "local line 1\nlocal line 2\n"
        with patch("builtins.open", mock_open(read_data=mock_file_data)) as mock_file:
            lines = list(LocalFileSystem.get_file_iterable(file_path))
            mock_file.assert_called_once_with(file_path, "r", encoding="utf-8")
            self.assertEqual(lines, ["local line 1\n", "local line 2\n"])

    def test_is_dir_true(self):
        dir_path = "somedirectory"
        with patch.object(Path, "exists", return_value=True), patch.object(Path, "is_dir", return_value=True):
            self.assertTrue(LocalFileSystem.is_dir(dir_path))

    def test_is_dir_false(self):
        dir_path = "somefile.txt"
        with patch.object(Path, "exists", return_value=True), patch.object(Path, "is_dir", return_value=False):
            self.assertFalse(LocalFileSystem.is_dir(dir_path))

    def test_is_dir_raises(self):
        dir_path = "nonexistent"
        with patch.object(Path, "exists", return_value=False):
            with self.assertRaises(RuntimeError):
                LocalFileSystem.is_dir(dir_path)

    def test_get_all_files_with_ext(self):
        dir_path = "somedirectory"
        extension = "txt"
        mock_file_list = [str(Path(dir_path) / f"file{i}.{extension}") for i in range(3)]
        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "glob", return_value=(Path(f) for f in mock_file_list)),
        ):
            files = list(LocalFileSystem.get_all_files_with_ext(dir_path, extension))
            self.assertEqual(files, mock_file_list)

    def test_get_all_files_with_ext_raises(self):
        dir_path = "nonexistentdirectory"
        with patch.object(Path, "exists", return_value=False):
            with self.assertRaises(RuntimeError):
                list(LocalFileSystem.get_all_files_with_ext(dir_path, "txt"))
