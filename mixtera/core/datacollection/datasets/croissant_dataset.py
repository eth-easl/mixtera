from pathlib import Path
from typing import Callable, Iterable, Optional

from mixtera.core.datacollection.datasets.dataset import Dataset, DatasetType
from mixtera.core.datacollection.index.parser import MetadataParser
from mixtera.network.connection import ServerConnection

#  TODO(#75): Implement CroissantDataset


class CroissantDataset(Dataset):
    type: DatasetType = DatasetType.CROISSANT_DATASET

    @staticmethod
    def build_file_index(loc: Path, metadata_parser: MetadataParser) -> None:
        raise NotImplementedError("CroissantDataset not yet supported.")

    @staticmethod
    def iterate_files(loc: str) -> Iterable[str]:
        raise NotImplementedError("CroissantDataset not yet supported.")

    @staticmethod
    def read_ranges_from_files(
        ranges_per_file: dict[str, list[tuple[int, int]]],
        parsing_func: Callable[[str], str],
        server_connection: Optional[ServerConnection],
    ) -> Iterable[str]:
        raise NotImplementedError()
