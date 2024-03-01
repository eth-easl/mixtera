from abc import ABC, abstractmethod
from typing import Generator, Iterable, Optional, Type
from contextlib import contextmanager
from mixtera.server import ServerConnection
from pathlib import Path

class AbstractFilesystem(ABC):
    type_id = 0

    @staticmethod
    def from_id(type_id: int) -> "Type[AbstractFilesystem]":
        """
        This method instantiates a filesystem from an integer type ID (e.g., stored in a DB).

        Args:
            type_id (int): Type ID that uniquely identifies the filesystem

        Returns:
            The class that belongs to the type_id.
        """
        if type_id < 1:
            raise RuntimeError(f"Invalid type id {type_id}")

        from mixtera.core.filesystem import (  # pylint: disable=import-outside-toplevel
            LocalFilesystem,
        )

        if type_id == LocalFilesystem.type_id:
            return LocalFilesystem

        raise NotImplementedError(f"type_id {type_id} not yet supported")

    @classmethod
    @abstractmethod
    def get_file_iterable(cls, file_path: str | Path, server_connection: Optional[ServerConnection] = None) -> Iterable[str]:
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
        pass

    @classmethod
    @contextmanager
    def open_file(cls, file_path: str | Path, server_connection: Optional[ServerConnection] = None) -> Generator[Iterable[str], None, None]:
        """
        Context manager to abstract the opening of files across different file systems.

        Args:
            file_path (str | Path): The path to the file to be opened.
            server_connection (Optional[ServerConnection]): If not None, an open ServerConnection to the
                Mixtera server from which the file is fetched instead. If None, the file is read from the 
                client directly.
        """
        try:
            iterable = cls.get_file_iterable(file_path, server_connection)
            yield iterable
        finally:
            cls.close_file(iterable, server_connection)
