import unittest
from pathlib import Path
from typing import Generator, Iterable, Optional
from unittest.mock import patch

from mixtera.core.filesystem import FileSystem
from mixtera.server import ServerConnection


class DummyFilesystem(FileSystem):

    @classmethod
    def get_file_iterable(cls, file_path: str, server_connection: Optional[ServerConnection] = None) -> Iterable[str]:
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
        lines = list(DummyFilesystem.get_file_iterable("dummy_path"))
        self.assertEqual(lines, ["line 1", "line 2"])

    def test_is_dir(self):
        self.assertTrue(DummyFilesystem.is_dir("dummy_path"))

    def test_get_all_files_with_ext(self):
        files = list(DummyFilesystem.get_all_files_with_ext(Path("/dummy_dir"), ".txt"))
        self.assertIn("/dummy_dir/file1.txt", files)
        self.assertIn("/dummy_dir/file2.txt", files)
