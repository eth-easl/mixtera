from pathlib import Path
from typing import Callable, Iterable, Optional, Type

from mixtera.core.datacollection.datasets.dataset import Dataset
from mixtera.core.datacollection.index.parser import MetadataParser
from mixtera.core.filesystem import FileSystem
from mixtera.server import ServerConnection


class CrossaintDataset(Dataset):
    type_id = 2

    @staticmethod
    def build_file_index(loc: Path, filesys_t: Type[FileSystem], metadata_parser: MetadataParser) -> None:
        raise NotImplementedError("CrossaintDataset not yet supported.")

    @staticmethod
    def iterate_files(loc: str, filesys_t: Type[FileSystem]) -> Iterable[str]:
        raise NotImplementedError("CrossaintDataset not yet supported.")

    @staticmethod
    def read_ranges_from_files(
        ranges_per_file: dict[str, list[tuple[int, int]]],
        filesys_t: Type[FileSystem],
        parsing_func: Callable[[str], str],
        server_connection: Optional[ServerConnection],
    ) -> Iterable[str]:
        raise NotImplementedError()
