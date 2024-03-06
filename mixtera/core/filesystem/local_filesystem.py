from pathlib import Path
from typing import TYPE_CHECKING, Generator, Iterable, Optional

from mixtera.core.filesystem import AbstractFilesystem

if TYPE_CHECKING:
    from mixtera.network.connection import ServerConnection


class LocalFilesystem(AbstractFilesystem):
    type_id = 1

    @classmethod
    def get_file_iterable(
        cls, file_path: str | Path, server_connection: Optional["ServerConnection"] = None
    ) -> Iterable[str]:
        if server_connection is not None:
            # TODO(create issue): We currently transfer the entire file, instead of parsing the ranges at server.
            # It is unclear what is better (local parsing vs at-client parsing)
            yield from server_connection.get_file_iterable(
                cls.type_id, file_path if isinstance(file_path, str) else str(file_path)
            )
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                yield from f

    @classmethod
    def is_dir(cls, path: str | Path) -> bool:
        dir_path = Path(path)

        if not dir_path.exists():
            raise RuntimeError(f"Path {path} does not exist.")

        return dir_path.is_dir()

    @classmethod
    def get_all_files_with_ext(cls, dir_path: str | Path, extension: str) -> Generator[str, None, None]:
        """
        Implements a generator that iterates over all files with a specific extension in a given directory.

        Args:
            dir_path (str | Path): The path in which all files checked for the extension.

        Returns:
            An iterable over the matching files.
        """
        n_dir_path = Path(dir_path)
        if not n_dir_path.exists():
            raise RuntimeError(f"Path {n_dir_path} does not exist!")

        for pth in n_dir_path.glob(f"*.{extension}"):
            yield str(pth)
