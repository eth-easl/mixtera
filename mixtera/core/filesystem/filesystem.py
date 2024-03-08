from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Iterable, Optional, Type

from mixtera.server import ServerConnection


class FileSystem(ABC):

    @staticmethod
    def from_path(file_path: str) -> "Type[FileSystem]":
        """
        This method instantiates a filesystem given a path (file or directory).

        The filesystem is inferred from the path prefix. Currently, only the local
        filesystem via a file:// or / prefix is supported.

        Args:
            file_path (str): File path to get the FileSystem for

        Returns:
            The Type[FileSystem] that belongs to the file_path.
        """
        file_path = str(file_path) if isinstance(file_path, Path) else file_path

        from mixtera.core.filesystem import LocalFileSystem  # pylint: disable=import-outside-toplevel

        if file_path.startswith("file://") or file_path.startswith("/"):
            return LocalFileSystem

        raise NotImplementedError(f"Cannot infer filesystem from path {file_path}")

    @staticmethod
    @contextmanager
    def open_file(
        file_path: str, server_connection: Optional[ServerConnection] = None
    ) -> Generator[Iterable[str], None, None]:
        """
        Context manager to abstract the opening of files across different file systems.

        Args:
            file_path (str): The path to the file to be opened.
            server_connection (Optional[ServerConnection]): If not None, an open ServerConnection to the
                Mixtera server from which the file is fetched instead. If None, the file is read from the
                client directly.
        """
        file_path = str(file_path) if isinstance(file_path, Path) else file_path

        if server_connection is not None:
            yield server_connection.get_file_iterable(file_path)
            return

        yield FileSystem.from_path(file_path).get_file_iterable(file_path)

    @classmethod
    @abstractmethod
    def get_file_iterable(cls, file_path: str) -> Iterable[str]:
        """
        Method to get an iterable of lines from a file that is stored on a file system.

        Args:
            file_path (str | Path): The path to the file to be opened.
            server_connection (Optional[ServerConnection]): If not None, an open ServerConnection to the
                Mixtera server from which the file is fetched instead. If None, the file is read from the
                client directly.

        Returns:
            An iterable of lines from the file.
        """
        raise NotImplementedError()

    @classmethod
    def is_dir(cls, path: str) -> bool:
        """
        Checks whether a given path is a directory or file.
        Since this is only run from a LocalDataCollection, this does not over a remote server interface.

        Args:
            path (str): The path to be checked.

        Returns:
            An boolean that is True if the path points to a directory.
        """
        path = str(path) if isinstance(path, Path) else path

        return FileSystem.from_path(path).is_dir(path)

    @classmethod
    def get_all_files_with_ext(cls, dir_path: str, extension: str) -> Generator[str, None, None]:
        """
        Implements a generator that iterates over all files with a specific extension in a given directory.
        Since this is only run from a LocalDataCollection, this does not over a remote server interface.

        Args:
            dir_path (str ): The path in which all files checked for the extension.

        Returns:
            An iterable over the matching files.
        """
        dir_path = str(dir_path) if isinstance(dir_path, Path) else dir_path

        yield from FileSystem.from_path(dir_path).get_all_files_with_ext(dir_path, extension)
