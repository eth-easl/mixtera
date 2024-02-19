from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from mixtera.core.datacollection import IndexType


class Dataset(ABC):
    type_id = 0

    @staticmethod
    def from_type_id(type_id: int) -> "Dataset":
        if type_id < 1:
            raise RuntimeError(f"Invalid type id {type_id}")

        # TODO(MaxiBoether): instantiate jsonl dataset here.

        raise NotImplementedError(f"type_id {type_id} not yet supported")

    @staticmethod
    @abstractmethod
    def build_file_index(loc: Path, dataset_id: int, file_id: int) -> IndexType:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def iterate_files(loc: str) -> Iterable[Path]:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def read_file(ranges_per_file: list[str]) -> Iterable[str]:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def read_ranges_from_files(ranges_per_file: dict[str, list[tuple[int, int]]]) -> Iterable[str]:
        raise NotImplementedError()
