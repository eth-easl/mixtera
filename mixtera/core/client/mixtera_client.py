import multiprocessing as mp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generator, Type

from mixtera.core.datacollection import PropertyType
from mixtera.core.datacollection.datasets import Dataset
from mixtera.core.datacollection.index.parser import MetadataParser
from mixtera.core.processing import ExecutionMode
from mixtera.core.query import Query, ResultChunk
from mixtera.core.query.mixture import Mixture
from mixtera.network.client.client_feedback import ClientFeedback

if TYPE_CHECKING:
    from mixtera.core.client.local import LocalStub
    from mixtera.core.client.server import ServerStub


@dataclass
class QueryExecutionArgs:
    mixture: Mixture
    dp_groups: int = 1
    nodes_per_group: int = 1
    num_workers: int = 1


@dataclass
class ResultStreamingArgs:
    job_id: str
    dp_group_id: int = 0
    node_id: int = 0
    worker_id: int = 0
    tunnel_via_server: bool = False
    chunk_reading_degree_of_parallelism: int = 1
    chunk_reading_per_window_mixture: bool = False
    chunk_reading_window_size: int = 128


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

        if len(args) == 1 and isinstance(args[0], tuple):
            args = args[0]

        if len(args) == 1 and isinstance(args[0], (str, Path)):
            from mixtera.core.client.local import LocalStub  # pylint:disable=import-outside-toplevel

            return object.__new__(LocalStub)

        if len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], int):
            from mixtera.core.client.server import ServerStub  # pylint:disable=import-outside-toplevel

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
        loc: str | Path,
        dtype: Type[Dataset],
        parsing_func: Callable[[str], str],
        metadata_parser_identifier: str,
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
            metadata_parser_identifier (str): the identifier of the metadata parser
                to be used for indexing. Can be registered using `register_metadata_parser`.

        Returns:
            Boolean indicating success.
        """

        raise NotImplementedError()

    @abstractmethod
    def register_metadata_parser(
        self,
        identifier: str,
        parser: Type[MetadataParser],
    ) -> bool:
        """
        This method registers a metadata parser in Mixtera.

        Args:
            identifier (str): The dataset identifier.
            parser (Type[MetadataParser]): The class object of the parser to register.
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
    def execute_query(self, query: Query, args: QueryExecutionArgs) -> bool:
        """
        Executes the query on the MixteraClient. Afterwards, result can be obtained using `stream_results`.

        Args:
            query (Query): The query to execute
            args (QueryExecutionArgs): The object encoding the execution arguments

        Returns:
            bool indicating success
        """

        raise NotImplementedError()

    def stream_results(self, args: ResultStreamingArgs) -> Generator[tuple[int, int, str], None, None]:
        """
        Given a job ID, returns the QueryResult object from which the result chunks can be obtained.
        Args:
            args (ResultStreamingArgs): The object encoding the streaming arguments
        Returns:
            A Generator over string samples.

        Raises:
            RuntimeError if query has not been executed.
        """
        for result_chunk in self._stream_result_chunks(args.job_id, args.dp_group_id, args.node_id, args.worker_id):
            result_chunk.configure_result_streaming(
                client=self,
                args=args,
            )
            yield from result_chunk

    @abstractmethod
    def _stream_result_chunks(
        self, job_id: str, dp_group_id: int, node_id: int, worker_id: int
    ) -> Generator[ResultChunk, None, None]:
        """
        Given a job ID, iterates over the result chunks.

        Args:
            job_id (str): The job ID to get the results for.
        Returns:
            A Generator over result chunks.

        Raises:
            RuntimeError if query has not been executed.
        """
        raise NotImplementedError()

    @abstractmethod
    def _get_result_metadata(
        self, job_id: str
    ) -> tuple[dict[int, Type[Dataset]], dict[int, Callable[[str], str]], dict[int, str]]:
        """
        Given a job ID, get metadata for the query result.

        Args:
            job_id (str): The job ID to get the results for.
        Returns:
            A tuple containing mappings to parse the results (dataset_type_dict, parsing_func_dict, file_path_dict)

        Raises:
            RuntimeError if query has not been executed.
        """
        raise NotImplementedError()

    @abstractmethod
    def checkpoint(
        self, job_id: str, dp_group_id: int, node_id: int, worker_status: list[int], server: bool = False
    ) -> str:
        """
        Initiates a checkpoint operation for the specified `job_id`. All nodes need to inform Mixtera
        about their current status via `worker_status`, which typically contains the indices of the samples
        each data loader worker is processing. The `MixteraTorchDataset` has the `worker_status` property
        for this purpose. You can use helper functions in `mixtera.utils.checkpointing` from your training loop.

        Args:
            job_id (str): The identifier of the job to checkpoint.
            dp_group_id (int): The data parallel group ID performing the checkpoint.
            node_id (int): The node ID where the checkpoint is being initiated.
            worker_status (list[int]): A list containing the current status of each data loader worker.
            server (bool, optional): If `True`, the checkpoint is being performed on the server side.
                                    Defaults to `False`.

        Returns:
            str: The identifier of the created checkpoint.

        Raises:
            RuntimeError: If the checkpoint operation fails.
        """
        raise NotImplementedError()

    @abstractmethod
    def checkpoint_completed(self, job_id: str, chkpnt_id: str, on_disk: bool) -> bool:
        """
        Checks whether the checkpoint `chkpnt_id` for `job_id` has been fully written.

        Args:
            job_id (str): The identifier of the job to check.
            chkpnt_id (str): The identifier of the checkpoint to verify.
            on_disk (bool): If `True`, returns `True` only if the checkpoint has been persisted to disk.
                            If `False`, returns `True` as soon as the checkpoint is stored in memory.

        Returns:
            bool: `True` if the checkpoint is complete (and written to disk if `on_disk` is `True`), `False` otherwise.

        Raises:
            RuntimeError: If there is an error checking the checkpoint status.
        """
        raise NotImplementedError()

    @abstractmethod
    def restore_checkpoint(self, job_id: str, chkpnt_id: str) -> None:
        """
        Restores the checkpoint `chkpnt_id` for `job_id`. After restoration, functions like
        `stream_results` can be called for this job to resume processing from the checkpoint.

        Args:
            job_id (str): The identifier of the job to restore.
            chkpnt_id (str): The identifier of the checkpoint to restore.

        Returns:
            None

        Raises:
            RuntimeError: If the checkpoint restoration fails.
        """
        raise NotImplementedError()

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
        degree_of_parallelism: int = 1,
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
            degree_of_parallelism (int): Degree of parallelism. How many processing units should be used in parallel.
                       Meaning depends on execution_mode
            data_only_on_primary (bool): If False, the processing units (may be remote machines)
                                         have access to the same paths as the primary.
        """

        raise NotImplementedError()

    @abstractmethod
    def send_feedback(self, job_id: str, feedback: ClientFeedback) -> bool:
        """
        This function sends the training feedback to the server, e.g., for the mixture schedule.
        """
        raise NotImplementedError()
