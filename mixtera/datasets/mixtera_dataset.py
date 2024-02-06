from abc import ABC, abstractmethod
from pathlib import Path

from mixtera.datasets import DatasetTypes
from mixtera.datasets.local import LocalMixteraDataset
from mixtera.datasets.remote import RemoteMixteraDataset


class MixteraDataset(ABC):

    @staticmethod
    def from_directory(directory: Path) -> LocalMixteraDataset:
        """
        Instantiates a LocalMixteraDataset from a directory.
        In this directory, Mixtera might create arbitrary files to manage metadata (e.g., a sqlite database).
        Information is persisted across instantiations in this database.
        New datasets can be added using the `register_dataset` function.

        Args:
            directory (Path): The directory where Mixtera stores its metadata files

        Returns:
            A LocalMixteraDataset instance.
        """
        if directory.exists():
            return LocalMixteraDataset(directory)

        raise RuntimeError(f"Directory {directory} does not exist.")

    @staticmethod
    def from_remote(endpoint: str) -> RemoteMixteraDataset:
        raise NotImplementedError("Remote datasets are not yet supported.")

    @abstractmethod
    def register_dataset(self, identifier: str, loc: str, dtype: DatasetTypes) -> bool:
        """
        This method registers a (sub)dataset in the MixteraDataset.

        Args:
            identifier (str): The dataset identifier.
            loc (str): The location where the (sub)dataset is stored.
                       For example, a path to a jsonl file.
            dtype (DatasetTypes): The type of the dataset.

        Returns:
            Boolean indicating success.
        """

        raise NotImplementedError()

    @abstractmethod
    def check_dataset_exists(self, identifier: str) -> bool:
        """
        Checks whether a (sub)dataset exists in the MixteraDataset

        Args:
            identifier (str): The identifier of the (sub)dataset

        Returns:
            Boolean indicating whtether the dataset exists.
        """

        raise NotImplementedError()
