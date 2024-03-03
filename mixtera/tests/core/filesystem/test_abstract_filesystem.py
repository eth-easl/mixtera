import unittest
from pathlib import Path
from typing import Generator, Iterable, Optional
from unittest.mock import patch

from mixtera.core.filesystem import AbstractFilesystem
from mixtera.server import ServerConnection


class DummyFilesystem(AbstractFilesystem):

    type_id = 1

    @classmethod
    def get_file_iterable(
        cls, file_path: str | Path, server_connection: Optional[ServerConnection] = None
    ) -> Iterable[str]:
        yield from ["line 1", "line 2"]

    @classmethod
    def is_dir(cls, path: str | Path) -> bool:
        return True

    @classmethod
    def get_all_files_with_ext(cls, dir_path: str | Path, extension: str) -> Generator[str, None, None]:
        if extension == ".txt":
            yield str(dir_path / "file1.txt")
            yield str(dir_path / "file2.txt")


class TestAbstractFilesystem(unittest.TestCase):

    def test_from_id_valid(self):
        # Assuming LocalFilesystem has type_id of 1 in its implementation
        with patch("mixtera.core.filesystem.LocalFilesystem") as mocked_local_filesystem:
            mocked_local_filesystem.type_id = 1
            filesystem_class = AbstractFilesystem.from_id(1)
            self.assertIs(filesystem_class, mocked_local_filesystem)

    def test_from_id_not_implemented(self):
        # Test an unregistered type_id
        with self.assertRaises(NotImplementedError):
            AbstractFilesystem.from_id(999)

    def test_from_id_invalid(self):
        with self.assertRaises(RuntimeError):
            AbstractFilesystem.from_id(0)

    def test_open_file(self):
        with patch.object(DummyFilesystem, "get_file_iterable") as mocked_get_file_iterable:
            mocked_get_file_iterable.return_value = ["line 1", "line 2"]
            with DummyFilesystem.open_file("dummy_path") as file_iter:
                self.assertEqual(list(file_iter), ["line 1", "line 2"])

    def test_get_file_iterable(self):
        lines = list(DummyFilesystem.get_file_iterable("dummy_path"))
        self.assertEqual(lines, ["line 1", "line 2"])

    def test_is_dir(self):
        self.assertTrue(DummyFilesystem.is_dir("dummy_path"))

    def test_get_all_files_with_ext(self):
        files = list(DummyFilesystem.get_all_files_with_ext(Path("/dummy_dir"), ".txt"))
        self.assertIn("/dummy_dir/file1.txt", files)
        self.assertIn("/dummy_dir/file2.txt", files)
