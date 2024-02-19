from pathlib import Path
from typing import Iterable

from mixtera.core.datacollection import IndexType
from mixtera.core.datacollection.datasets.dataset import Dataset


class CrossaintDataset(Dataset):
    type_id = 2

    @staticmethod
    def build_file_index(loc: Path, dataset_id: int, file_id: int) -> IndexType:
        raise NotImplementedError("CrossaintDataset not yet supported.")

    @staticmethod
    def iterate_files(loc: str) -> Iterable[Path]:
        raise NotImplementedError("CrossaintDataset not yet supported.")

    @staticmethod
    def read_file(ranges_per_file: list[str]) -> Iterable[str]:
        raise NotImplementedError("CrossaintDataset not yet supported.")

    @staticmethod
    def read_ranges_from_files(ranges_per_file: dict[str, list[tuple[int, int]]]) -> Iterable[str]:
        raise NotImplementedError("CrossaintDataset not yet supported.")
