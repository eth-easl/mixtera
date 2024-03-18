import multiprocessing as mp
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generator, Type

from mixtera.core.datacollection import PropertyType
from mixtera.core.datacollection.datasets import Dataset
from mixtera.core.datacollection.index import IndexType
from mixtera.core.processing import ExecutionMode
from mixtera.core.query import Query

if TYPE_CHECKING:
    from mixtera.core.client.local import LocalStub
    from mixtera.core.client.server import ServerStub


class MixteraClient(ABC):

    def __new__(cls, *args: Any) -> "MixteraClient":
        """
        Meta-function to dispatch calls to the constructor of MixteraClient to the ServerStub
        or LocalStub.

        If you are facing pylint issues due to instantiation of abstract classes, consider using
        from_directory/from_remote instead.
        """
        if not args and mp.current_process().name != "MainProcess":
            # We are in a spawned child process, so we might be unpickling
            # Allow creation without args to support the unpickling process
            # Leads to runtime errors on macOS/Windows otherwise.
            return object.__new__(cls)

        from mixtera.core.client.local import LocalStub  # pylint: disable=import-outside-toplevel
        from mixtera.core.client.server import ServerStub  # pylint: disable=import-outside-toplevel

        if len(args) == 1:
            param = args[0]
            if isinstance(param, (str, Path)):
                return object.__new__(LocalStub)
            if isinstance(param, tuple):
                if len(param) == 2:
                    return object.__new__(ServerStub)

        if len(args) == 2:
            return object.__new__(ServerStub)

        raise ValueError(f"Invalid parameter type(s): {args}. Please use from_directory/from_server functions.")

    @staticmethod
    def from_directory(directory: Path | str) -> "LocalStub":
        """
        Instantiates a LocalStub from a directory.
        In this directory, Mixtera might create arbitrary files to manage metadata (e.g., a sqlite database).
        Information is persisted across instantiations in this database.
        New datasets can be added using the `register_dataset` function.

        Args:
            directory (Path or str): The directory where Mixtera stores its metadata files

        Returns:
            A LocalStub instance.
        """
        # Local import to avoid circular dependency
        from mixtera.core.client.local import LocalStub  # pylint: disable=import-outside-toplevel

        return LocalStub(directory)

    @staticmethod
    def from_remote(host: str, port: int) -> "ServerStub":
        """
        Instantiates a ServerStub from a host address and port.

        Args:
            host (str): The host address of the Mixtera server
            port (int): The port of the Mixtera server

        Returns:
            A RemoteDataCollection instance.
        """

        # Local import to avoid circular dependency
        from mixtera.core.client.server import ServerStub  # pylint: disable=import-outside-toplevel

        return ServerStub(host, port)

    @abstractmethod
    def register_dataset(
        self,
        identifier: str,
        loc: str,
        dtype: Type[Dataset],
        parsing_func: Callable[[str], str],
        metadata_parser_type: str,
    ) -> bool:
        """
        This method registers a dataset in Mixtera.

        Args:
            identifier (str): The dataset identifier.
            loc (str): The location where the dataset is stored.
                       For example, a path to a directory of jsonl files.
            dtype (Type[Dataset]): The type of the dataset.
            parsing_func (Callable[[str], str]): A function that given one "base unit"
                of a file in the data set extracts the actual sample. The meaning depends
                on the dataset type at hand. For example, for the JSONLDataset, every line
                is processed with this function and it can be used to extract the actual
                payload out of the metadata.
            metadata_parser_type: the name of the metadata parser to be used for indexing

        Returns:
            Boolean indicating success.
        """

        raise NotImplementedError()

    @abstractmethod
    def check_dataset_exists(self, identifier: str) -> bool:
        """
        Check whether dataset is registered in Mixtera.

        Args:
            identifier (str): The identifier of the dataset

        Returns:
            Boolean indicating whtether the dataset exists.
        """

        raise NotImplementedError()

    @abstractmethod
    def list_datasets(self) -> list[str]:
        """
        Lists all registered datasets.

        Args:
            identifier (str): The identifier of the (sub)dataset

        Returns:
            List of dataset identifiers.
        """

        raise NotImplementedError()

    @abstractmethod
    def remove_dataset(self, identifier: str) -> bool:
        """
        Removes (unregisters) a dataset from the Mixtera

        Args:
            identifier (str): The identifier of the dataset

        Returns:
            Boolean indicating success of the operation.
        """

        raise NotImplementedError()

    @abstractmethod
    def execute_query(self, query: Query, chunk_size: int) -> bool:
        """
        Executes the query on the MixteraClient. Afterwards, result can be obtained using `stream_results`.

        Args:
            query (Query): The query to execute.
            chunk_size (int): chunk_size is used to set the size of `subresults` in the QueryResult object.
                Defaults to 1. When iterating over a :py:class:`QueryResult`
                object, the results are yielded in chunks of size `chunk_size`. Relevant for throughput
                optimization.
        Returns:
            bool indicating success
        """

        raise NotImplementedError()

    def stream_results(self, training_id: str, tunnel_via_server: bool) -> Generator[str, None, None]:
        """
        Given a training ID, returns the QueryResult object from which the result chunks can be obtained.
        Args:
            training_id (str): The training ID to get the results for.
            tunnel_via_server (bool): If true, samples are streamed via the Mixtera server.
        Returns:
            A Generator over string samples.

        Raises:
            RuntimeError if query has not been executed. # TODO (MaxiBoether): dont forget this!
        """
        result_metadata = self._get_result_metadata(training_id)
        for result_chunk in self._stream_result_chunks(training_id):
            # TODO(): When implementing the new sampling on the ResultChunk,
            # the ResultChunk class should offer an iterator instead.
            yield from self._iterate_result_chunk(result_chunk, *result_metadata, tunnel_via_server=tunnel_via_server)

    @abstractmethod
    def _stream_result_chunks(self, training_id: str) -> Generator[IndexType, None, None]:
        """
        Given a training ID, iterates over the result chunks.

        Args:
            training_id (str): The training ID to get the results for.
        Returns:
            A Generator over result chunks.

        Raises:
            RuntimeError if query has not been executed. # TODO (MaxiBoether): dont forget this!
        """
        raise NotImplementedError()

    @abstractmethod
    def _get_result_metadata(
        self, training_id: str
    ) -> tuple[dict[int, Type[Dataset]], dict[int, Callable[[str], str]], dict[int, str]]:
        """
        Given a training ID, get metadata for the query result.

        Args:
            training_id (str): The training ID to get the results for.
        Returns:
            A tuple containing mappings to parse the results (dataset_type_dict, parsing_func_dict, file_path_dict)

        Raises:
            RuntimeError if query has not been executed. # TODO (MaxiBoether): dont forget this!
        """
        raise NotImplementedError()

    def _iterate_result_chunk(
        self,
        result_chunk: IndexType,
        dataset_type_dict: dict[int, Type[Dataset]],
        parsing_func_dict: dict[int, Callable[[str], str]],
        file_path_dict: dict[int, str],
        tunnel_via_server: bool = False,
    ) -> Generator[str, None, None]:
        """
        Given a result chunk, iterates over the samples.

        Args:
            result_chunk (IndexType): The result chunk object.
            dataset_type_dict (dict): A mapping from dataset ID to dataset type.
            parsing_func_dict (dict): A mapping from dataset ID to parsing function.
            file_path_dict (dict): A mapping from file ID to file path.
            tunnel_via_server (bool): If true, samples are streamed via the Mixtera server.

        Returns:
            A Generator of samples.
        """
        # TODO(create issue): Currently, the result chunks are IndexType,
        # but they should offer their own class with an iterator over samples.
        # This should sample correctly from the chunk. Then, there is no need for this function anymore.

        from mixtera.core.client.server import ServerStub  # pylint: disable=import-outside-toplevel

        server_connection = None
        if tunnel_via_server:
            if isinstance(self, ServerStub):
                server_connection = self._server_connection
            else:
                raise RuntimeError(
                    "Currently, tunneling samples via the server is only supported when using a ServerStub."
                )

        for _, property_dict in result_chunk._index.items():
            for _, val_dict in property_dict.items():
                for did, file_dict in val_dict.items():
                    filename_dict = {file_path_dict[file_id]: file_ranges for file_id, file_ranges in file_dict.items()}
                    yield from dataset_type_dict[did].read_ranges_from_files(
                        filename_dict, parsing_func_dict[did], server_connection
                    )

    @abstractmethod
    def is_remote(self) -> bool:
        """
        Checks whether the Mixtera client object at hand uses a server or local MDC.

        Returns:
            A bool that is true if connected to a server.
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
        This function extends the Mixtera index with a new property that is calculated per sample in the collection.

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
