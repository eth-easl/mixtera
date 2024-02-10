from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, List

from mixtera.core.datacollection.dataset_types import DatasetTypes

if TYPE_CHECKING:
    from mixtera.core.datacollection.local import LocalDataCollection
    from mixtera.core.datacollection.remote import RemoteDataCollection


class MixteraDataCollection(ABC):

    @staticmethod
    def from_directory(directory: Path | str) -> "LocalDataCollection":
        """
        Instantiates a LocalCollection from a directory.
        In this directory, Mixtera might create arbitrary files to manage metadata (e.g., a sqlite database).
        Information is persisted across instantiations in this database.
        New datasets can be added using the `register_dataset` function.

        Args:
            directory (Path): The directory where Mixtera stores its metadata files

        Returns:
            A LocalDataCollection instance.
        """
        # Local import to avoid circular dependency
        from mixtera.core.datacollection.local import LocalDataCollection  # pylint: disable=import-outside-toplevel

        if isinstance(directory, str):
            dir_path = Path(directory)
        else:
            dir_path = directory

        if dir_path.exists():
            return LocalDataCollection(dir_path)

        raise RuntimeError(f"Directory {dir_path} does not exist.")

    @staticmethod
    def from_remote(endpoint: str) -> "RemoteDataCollection":
        raise NotImplementedError("Remote datasets are not yet supported.")

    @abstractmethod
    def register_dataset(self, identifier: str, loc: str, dtype: DatasetTypes) -> bool:
        """
        This method registers a dataset in the MixteraDataCollection.

        Args:
            identifier (str): The dataset identifier.
            loc (str): The location where the dataset is stored.
                       For example, a path to a directory of jsonl files.
            dtype (DatasetTypes): The type of the dataset.

        Returns:
            Boolean indicating success.
        """

        raise NotImplementedError()

    @abstractmethod
    def check_dataset_exists(self, identifier: str) -> bool:
        """
        Checks whether a dataset exists in the MixteraDataCollection

        Args:
            identifier (str): The identifier of the dataset

        Returns:
            Boolean indicating whtether the dataset exists.
        """

        raise NotImplementedError()

    @abstractmethod
    def list_datasets(self) -> List[str]:
        """
        Lists all datasets that are part of the MixteraDataCollection

        Args:
            identifier (str): The identifier of the (sub)dataset

        Returns:
            List of dataset identifiers.
        """

        raise NotImplementedError()

    @abstractmethod
    def remove_dataset(self, identifier: str) -> bool:
        """
        Removes a dataset from the MixteraDataCollection

        Args:
            identifier (str): The identifier of the dataset

        Returns:
            Boolean indicating success of the operation.
        """

        raise NotImplementedError()
