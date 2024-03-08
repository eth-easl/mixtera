from pathlib import Path
from typing import Generator, Iterable, Optional

from mixtera.core.filesystem import FileSystem
from mixtera.server import ServerConnection


class LocalFilesystem(FileSystem):
    @classmethod
    def get_file_iterable(cls, file_path: str, server_connection: Optional[ServerConnection] = None) -> Iterable[str]:
        if server_connection is not None:
            yield from server_connection.get_file_iterable(file_path)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                yield from f

    @classmethod
    def is_dir(cls, path: str) -> bool:
        dir_path = Path(path)

        if not dir_path.exists():
            raise RuntimeError(f"Path {path} does not exist.")

        return dir_path.is_dir()

    @classmethod
    def get_all_files_with_ext(cls, dir_path: str, extension: str) -> Generator[str, None, None]:
        """
        Implements a generator that iterates over all files with a specific extension in a given directory.

        Args:
            dir_path (str): The path in which all files checked for the extension.

        Returns:
            An iterable over the matching files.
        """
        n_dir_path = Path(dir_path)
        if not n_dir_path.exists():
            raise RuntimeError(f"Path {n_dir_path} does not exist!")

        for pth in n_dir_path.glob(f"*.{extension}"):
            yield str(pth)
