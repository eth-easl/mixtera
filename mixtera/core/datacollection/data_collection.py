from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, List, Optional, Type

from mixtera.core.datacollection.datasets import Dataset
from mixtera.core.processing import ExecutionMode

if TYPE_CHECKING:
    from mixtera.core.datacollection import IndexType, PropertyType
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
    def register_dataset(
        self, identifier: str, loc: str, dtype: Type[Dataset], parsing_func: Callable[[str], str]
    ) -> bool:
        """
        This method registers a dataset in the MixteraDataCollection.

        Args:
            identifier (str): The dataset identifier.
            loc (str): The location where the dataset is stored.
                       For example, a path to a directory of jsonl files.
            dtype (DatasetTypes): The type of the dataset.
            parsing_func (Callable[[str], str]): A function that given one "base unit"
                of a file in the data set extracts the actual sample. The meaning depends
                on the dataset type at hand. For example, for the JSONLDataset, every line
                is processed with this function and it can be used to extract the actual
                payload out of the metadata.

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

    @abstractmethod
    def get_samples_from_ranges(
        self, ranges_per_dataset_and_file: dict[int, dict[int, list[tuple[int, int]]]]
    ) -> Iterable[str]:
        """
        Given a list of ranges for each file in each datasets, returns an Iterable over the samples.

        Args:
            ranges_per_dataset_and_file (dict[int, dict[int, list[tuple[int, int]]]]): Dict that
                maps each dataset ID to another dict containing file IDs as keys. This dict then
                contains the ranges as values.

        Returns:
            Iterable over the samples.
        """
        raise NotImplementedError()

    @abstractmethod
    def add_property(
        self,
        property_name: str,
        setup_func: Callable,
        calc_func: Callable,
        execution_mode: ExecutionMode,
        property_type: "PropertyType",
        min_val: float = 0.0,
        max_val: float = 1,
        num_buckets: int = 10,
        batch_size: int = 1,
        dop: int = 1,
        data_only_on_primary: bool = True,
    ) -> None:
        """
        This function extends the index with a new property that is calculated per sample in the collection.

        This can, for example, be some classification result (e.g., toxicity score or a language classifier).
        We can then use this new property in subsequent queries to the data.

        Args:
            property_name (str): The name of the new property that is added to the Mixtera index
            setup_func (Callable): Function that performs setup (e.g., load model).
                                   It is passed an instance of a class to put attributes on.
            calc_func (Callable): The function that given a batch of data calculates a numerical or categorical value.
                                  It has access to the class that was prepared by the setup_func.
            execution_mode (ExecutionMode): How to execute the function, i.e., on Ray or locally
            property_type (PropertyType): Whether it is a categorical or numerical property
            min_val (float): Optional value for numerical properties specifying the min value the property can take
            max_val (float): Optional value for numerical properties specifying the max value the property can take
            num_buckets (int): The number of buckets for numeritcal properties
            batch_size (int): Size of one batch passed to one processing instance
            dop (int): Degree of parallelism. How many processing units should be used in parallel.
                       Meaning depends on execution_mode
            data_only_on_primary (bool): If False, the processing units (may be remote machines)
                                         have access to the same paths as the primary.
        """

        raise NotImplementedError()

    @abstractmethod
    def get_index(self, property_name: Optional[str] = None) -> "IndexType":
        """
        This function returns the index of the MixteraDataCollection.

        Args:
            property_name (Optional[str], optional): The name of the property to query.
                If not provided, all properties are returned.
        """
        raise NotImplementedError()
