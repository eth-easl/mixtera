from typing import Iterable, Optional
from mixtera.core.filesystem import AbstractFilesystem
from mixtera.server import ServerConnection
from pathlib import Path

class LocalFilesystem(AbstractFilesystem):
    filesys_id = 1

    @classmethod
    def get_file_iterable(cls, file_path: str | Path, server_connection: Optional[ServerConnection]) -> Iterable[str]:
        if server_connection is not None:
            yield from server_connection.get_file_iterable(cls.filesys_id, file_path if isinstance(file_path, str) else str(file_path))
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                yield from f

    @classmethod
    def close_file(cls, iterable: Iterable[str]):
        # For the local filesystem, the file is automatically closed after exiting the 'with' block.
        # Therefore, we don't need to do anything here.
        pass
