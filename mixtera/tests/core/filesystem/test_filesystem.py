import unittest
from pathlib import Path
from typing import Generator, Iterable
from unittest.mock import MagicMock, patch

from mixtera.core.filesystem import FileSystem
from mixtera.network.connection import ServerConnection


class DummyFileSystem(FileSystem):
    @classmethod
    def get_file_iterable(cls, file_path: str) -> Iterable[str]:
        yield from ["line 1", "line 2"]

    @classmethod
    def is_dir(cls, path: str) -> bool:
        return True

    @classmethod
    def get_all_files_with_ext(cls, dir_path: str, extension: str) -> Generator[str, None, None]:
        if extension == ".txt":
            yield str(dir_path / "file1.txt")
            yield str(dir_path / "file2.txt")


class TestFileSystem(unittest.TestCase):
    def setUp(self):
        self.mock_server_connection = MagicMock(spec=ServerConnection)
        self.mock_server_connection.get_file_iterable.return_value = ["server line 1", "server line 2"]

    def tearDown(self):
        self.mock_server_connection = None

    def test_from_path(self):
        with patch("mixtera.core.filesystem.LocalFileSystem") as mocked_local_filesystem:
            filesystem_class = FileSystem.from_path("file://test.txt")
            self.assertIs(filesystem_class, mocked_local_filesystem)
            filesystem_class = FileSystem.from_path("/test.txt")
            self.assertIs(filesystem_class, mocked_local_filesystem)

    def test_from_id_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            FileSystem.from_path("test.txt")

    def test_get_file_iterable(self):
        lines = list(DummyFileSystem.get_file_iterable("dummy_path"))
        self.assertEqual(lines, ["line 1", "line 2"])

    def test_is_dir(self):
        self.assertTrue(DummyFileSystem.is_dir("dummy_path"))

    def test_get_all_files_with_ext(self):
        files = list(DummyFileSystem.get_all_files_with_ext(Path("/dummy_dir"), ".txt"))
        self.assertIn("/dummy_dir/file1.txt", files)
        self.assertIn("/dummy_dir/file2.txt", files)

    def test_open_file_with_server_connection(self):
        file_path = "testfile.txt"

        with FileSystem.open_file(file_path, self.mock_server_connection) as file:
            self.assertEqual(list(file), ["server line 1", "server line 2"])
            self.mock_server_connection.get_file_iterable.assert_called_once_with(file_path)
