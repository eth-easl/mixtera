import asyncio
from typing import TYPE_CHECKING, Any, Callable, Generator, Iterable, Optional, Type

from loguru import logger
from mixtera.core.datacollection import PropertyType
from mixtera.core.datacollection.datasets.dataset import Dataset
from mixtera.core.datacollection.index.parser import MetadataParser
from mixtera.core.processing.execution_mode import ExecutionMode
from mixtera.network import NUM_BYTES_FOR_IDENTIFIERS, NUM_BYTES_FOR_SIZES
from mixtera.network.network_utils import (
    read_int,
    read_pickeled_object,
    read_utf8_string,
    write_float,
    write_int,
    write_pickeled_object,
    write_utf8_string,
)
from mixtera.network.server_task import ServerTask
from mixtera.utils import run_async_until_complete

if TYPE_CHECKING:
    from mixtera.core.datacollection.index import ChunkerIndex
    from mixtera.core.query import Mixture, Query


class ServerConnection:
    """
    Provides an synchronous interface for connecting to a server, executing queries,
    fetching files, and streaming result chunks. This class handles asynchronous network
    communication details and exposes synchronous, higher-level methods to interact
    with the server.
    """

    def __init__(self, host: str, port: int) -> None:
        """
        Initializes the ServerConnection instance with the given server address.

        Args:
            host (str): The host address of the server.
            port (int): The port number of the server.
        """
        self._host = host
        self._port = port

    async def _fetch_file(self, file_path: str) -> Optional[str]:
        """
        Asynchronously fetches the content of a file from the server.

        Args:
            file_path (str): The path of the file to be fetched from the server.

        Returns:
            The content of the file as a string, or None if the connection fails.
        """
        reader, writer = await self._connect_to_server()

        if reader is None or writer is None:
            return None

        await write_int(int(ServerTask.READ_FILE), NUM_BYTES_FOR_IDENTIFIERS, writer)
        await write_utf8_string(file_path, NUM_BYTES_FOR_IDENTIFIERS, writer)

        return await read_utf8_string(NUM_BYTES_FOR_SIZES, reader)

    def get_file_iterable(self, file_path: str) -> Iterable[str]:
        """
        Provides an iterable over the lines of a file fetched from the server.

        Args:
            file_path (str): The path of the file to be fetched from the server.

        Yields:
            The lines of the file as an iterable, line by line.
            An empty iterator if the connection fails.
        """
        if (lines := run_async_until_complete(self._fetch_file(file_path))) is None:
            return

        yield from lines.split("\n")

    async def _connect_to_server(
        self, max_retries: int = 5, retry_delay: int = 1
    ) -> tuple[Optional[asyncio.StreamReader], Optional[asyncio.StreamWriter]]:
        """
        Asynchronously establishes a connection to the server, retrying upon failure up to a maximum number of attempts.

        Args:
            max_retries (int): The maximum number of connection attempts. Defaults to 5.
            retry_delay (int): The delay in seconds between connection attempts. Defaults to 1.

        Returns:
            A tuple containing the StreamReader and StreamWriter objects if the connection is successful,
            or (None, None) if the connection ultimately fails after the maximum number of retries.
        """
        for attempt in range(1, max_retries + 1):
            try:
                reader, writer = await asyncio.wait_for(asyncio.open_connection(self._host, self._port), timeout=5.0)
                return reader, writer
            except asyncio.TimeoutError:
                logger.error(f"Connection to {self._host}:{self._port} timed out (attempt {attempt}/{max_retries}).")
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error(
                    "Failed to connect to"
                    + f"{self._host}:{self._port}. Is the server running? (attempt {attempt}/{max_retries}):{e}"
                )

            if attempt < max_retries:
                await asyncio.sleep(retry_delay)
            else:
                logger.error(
                    "Maximum number of connection attempts"
                    + f"({max_retries}) reached. Unable to connect to {self._host}:{self._port}."
                )

        return None, None

    async def _execute_query(self, query: "Query", mixture: "Mixture") -> bool:
        """
        Asynchronously executes a query on the server and receives a confirmation of success.

        Args:
            query (Query): The query object to be executed.
            mixture: mixture object required by for chunking the result

        Returns:
            A boolean indicating whether the query was successfully registered with the server.
        """
        reader, writer = await self._connect_to_server()

        if reader is None or writer is None:
            return False

        # Announce we want to register a query
        await write_int(int(ServerTask.REGISTER_QUERY), NUM_BYTES_FOR_IDENTIFIERS, writer)

        # Announce mixture
        await write_pickeled_object(mixture, NUM_BYTES_FOR_SIZES, writer)

        # Announce query
        await write_pickeled_object(query, NUM_BYTES_FOR_SIZES, writer)

        success = bool(await read_int(NUM_BYTES_FOR_IDENTIFIERS, reader))
        logger.debug(f"Got success = {success} from server.")
        return success

    def execute_query(self, query: "Query", mixture: "Mixture") -> bool:
        """
        Executes a query on the server and returns whether it was successful.

        Args:
            query (Query): The query object to be executed.
            mixture: Mixture object required for chunking.

        Returns:
            A boolean indicating whether the query was successfully registered with the server.
        """
        success = run_async_until_complete(self._execute_query(query, mixture))
        return success

    async def _get_query_result_meta(self, job_id: str) -> Optional[dict]:
        """
        Asynchronously retrieves metadata about the query result from the server.

        Args:
            job_id (str): The identifier of the job for which result metadata is requested.

        Returns:
            A dictionary containing metadata about the query result, or None if the connection fails.
        """
        reader, writer = await self._connect_to_server()

        if reader is None or writer is None:
            return None

        # Announce we want to get the query meta result
        await write_int(int(ServerTask.GET_META_RESULT), NUM_BYTES_FOR_IDENTIFIERS, writer)

        # Announce job ID
        await write_utf8_string(job_id, NUM_BYTES_FOR_IDENTIFIERS, writer)

        # Get meta object
        return await read_pickeled_object(NUM_BYTES_FOR_SIZES, reader)

    # TODO(#35): Use some ResultChunk type
    async def _get_next_result(self, job_id: str) -> Optional["ChunkerIndex"]:
        """
        Asynchronously retrieves the next result chunk of a query from the server.

        Args:
            job_id (str): The identifier of the job for which the next result chunk is requested.

        Returns:
            An ChunkerIndex object representing the next result chunk,
            or None if there are no more results or the connection fails.
        """
        reader, writer = await self._connect_to_server()

        if reader is None or writer is None:
            return None

        # Announce we want to get a result chunk
        await write_int(int(ServerTask.GET_NEXT_RESULT_CHUNK), NUM_BYTES_FOR_IDENTIFIERS, writer)

        # Announce job ID
        await write_utf8_string(job_id, NUM_BYTES_FOR_IDENTIFIERS, writer)

        # Get meta object
        return await read_pickeled_object(NUM_BYTES_FOR_SIZES, reader)

    def _stream_result_chunks(self, job_id: str) -> Generator["ChunkerIndex", None, None]:
        """
        Streams the result chunks of a query job from the server.

        Args:
            job_id (str): The identifier of the job whose result chunks are to be streamed.

        Yields:
            ChunkerIndex objects, each representing a chunk of the query results.
        """
        # TODO(#62): We might want to prefetch here
        while (next_result := run_async_until_complete(self._get_next_result(job_id))) is not None:
            yield next_result

    def get_result_metadata(
        self, job_id: str
    ) -> tuple[dict[int, Any], dict[int, Callable[[str], str]], dict[int, str]]:
        """
        Retrieves the metadata associated with the result chunks of a query job.

        Args:
            job_id (str): The identifier of the job whose result metadata is to be retrieved.

        Raises:
            RuntimeError: If an error occurs while fetching the metadata from the server.

        Returns:
            A tuple containing three dictionaries:
            - The dataset types by their index
            - Parsing functions by their index
            - File paths by their index
        """
        if (meta := run_async_until_complete(self._get_query_result_meta(job_id))) is None:
            raise RuntimeError("Error while fetching meta results")

        return meta["dataset_type"], meta["parsing_func"], meta["file_path"]

    def register_dataset(
        self,
        identifier: str,
        loc: str,
        dtype: Type["Dataset"],
        parsing_func: Callable[[str], str],
        metadata_parser_identifier: str,
    ) -> bool:
        """
        Registers a dataset with the server.

        Args:
            identifier (str): The identifier of the dataset.
            loc (str): The location of the dataset.
            dtype (Type[Dataset]): The dataset class to be registered.
            parsing_func (Callable[[str], str]): The parsing function to be registered.
            metadata_parser_identifier (str): The identifier of the metadata parser.

        Returns:
            A boolean indicating whether the dataset was successfully registered with the server.
        """
        return run_async_until_complete(
            self._register_dataset(identifier, loc, dtype, parsing_func, metadata_parser_identifier)
        )

    async def _register_dataset(
        self,
        identifier: str,
        loc: str,
        dtype: Type["Dataset"],
        parsing_func: Callable[[str], str],
        metadata_parser_identifier: str,
    ) -> bool:
        """
        Asynchronously registers a dataset with the server.

        Args:
            identifier (str): The identifier of the dataset.
            loc (str): The location of the dataset.
            dtype (Type[Dataset]): The dataset class to be registered.
            parsing_func (Callable[[str], str]): The parsing function to be registered.
            metadata_parser_identifier (str): The identifier of the metadata parser.

        Returns:
            A boolean indicating whether the dataset was successfully registered with the server.
        """
        reader, writer = await self._connect_to_server()

        if reader is None or writer is None:
            return False

        # Announce we want to register a dataset
        await write_int(int(ServerTask.REGISTER_DATASET), NUM_BYTES_FOR_IDENTIFIERS, writer)

        # Announce dataset identifier
        await write_utf8_string(identifier, NUM_BYTES_FOR_IDENTIFIERS, writer)

        # Announce dataset location
        await write_utf8_string(loc, NUM_BYTES_FOR_IDENTIFIERS, writer)

        # Announce dataset class
        await write_int(dtype.type.value, NUM_BYTES_FOR_IDENTIFIERS, writer)

        # Announce parsing function
        await write_pickeled_object(parsing_func, NUM_BYTES_FOR_SIZES, writer)

        # Announce metadata parser identifier
        await write_utf8_string(metadata_parser_identifier, NUM_BYTES_FOR_IDENTIFIERS, writer)

        return bool(await read_int(NUM_BYTES_FOR_IDENTIFIERS, reader))

    def register_metadata_parser(self, identifier: str, parser: Type["MetadataParser"]) -> None:
        """
        Registers a metadata parser with the server.

        Args:
            identifier (str): The identifier of the metadata parser.
            parser (Type[MetadataParser]): The parser class to be registered.
        """
        run_async_until_complete(self._register_metadata_parser(identifier, parser))

    async def _register_metadata_parser(self, identifier: str, parser: Type["MetadataParser"]) -> None:
        """
        Asynchronously registers a metadata parser with the server.

        Args:
            identifier (str): The identifier of the metadata parser.
            parser (Type[MetadataParser]): The parser class to be registered.
        """
        reader, writer = await self._connect_to_server()

        if reader is None or writer is None:
            return

        # Announce we want to register a metadata parser
        await write_int(int(ServerTask.REGISTER_METADATA_PARSER), NUM_BYTES_FOR_IDENTIFIERS, writer)

        # Announce metadata parser identifier
        await write_utf8_string(identifier, NUM_BYTES_FOR_IDENTIFIERS, writer)

        # Announce metadata parser class
        await write_int(parser.type.value, NUM_BYTES_FOR_IDENTIFIERS, writer)

    def check_dataset_exists(self, identifier: str) -> bool:
        """
        Checks whether a dataset with the given identifier exists on the server.

        Args:
            identifier (str): The identifier of the dataset to check.

        Returns:
            A boolean indicating whether the dataset exists on the server.
        """
        return run_async_until_complete(self._check_dataset_exists(identifier))

    async def _check_dataset_exists(self, identifier: str) -> bool:
        """
        Asynchronously checks whether a dataset with the given identifier exists on the server.

        Args:
            identifier (str): The identifier of the dataset to check.

        Returns:
            A boolean indicating whether the dataset exists on the server.
        """
        reader, writer = await self._connect_to_server()

        if reader is None or writer is None:
            return False

        # Announce we want to check if a dataset exists
        await write_int(int(ServerTask.CHECK_DATASET_EXISTS), NUM_BYTES_FOR_IDENTIFIERS, writer)

        # Announce dataset identifier
        await write_utf8_string(identifier, NUM_BYTES_FOR_IDENTIFIERS, writer)

        return bool(await read_int(NUM_BYTES_FOR_IDENTIFIERS, reader))

    def list_datasets(self) -> list[str]:
        """
        Lists the datasets available on the server.

        Returns:
            A list of strings, each representing a dataset identifier.
        """
        return run_async_until_complete(self._list_datasets())

    async def _list_datasets(self) -> list[str]:
        """
        Asynchronously lists the datasets available on the server.

        Returns:
            A list of strings, each representing a dataset identifier.
        """
        reader, writer = await self._connect_to_server()

        if reader is None or writer is None:
            return []

        # Announce we want to list datasets
        await write_int(int(ServerTask.LIST_DATASETS), NUM_BYTES_FOR_IDENTIFIERS, writer)

        return await read_pickeled_object(NUM_BYTES_FOR_SIZES, reader)

    def remove_dataset(self, identifier: str) -> bool:
        """
        Removes a dataset from the server.

        Args:
            identifier (str): The identifier of the dataset to be removed.

        Returns:
            A boolean indicating whether the dataset was successfully removed from the server.
        """
        return run_async_until_complete(self._remove_dataset(identifier))

    async def _remove_dataset(self, identifier: str) -> bool:
        """
        Asynchronously removes a dataset from the server.

        Args:
            identifier (str): The identifier of the dataset to be removed.

        Returns:
            A boolean indicating whether the dataset was successfully removed from the server.
        """
        reader, writer = await self._connect_to_server()

        if reader is None or writer is None:
            return False

        # Announce we want to remove a dataset
        await write_int(int(ServerTask.REMOVE_DATASET), NUM_BYTES_FOR_IDENTIFIERS, writer)

        # Announce dataset identifier
        await write_utf8_string(identifier, NUM_BYTES_FOR_IDENTIFIERS, writer)

        return bool(await read_int(NUM_BYTES_FOR_IDENTIFIERS, reader))

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
        Adds a property to the server.

        Args:
            property_name (str): The name of the property.
            setup_func (Callable): The setup function for the property.
            calc_func (Callable): The calculation function for the property.
            execution_mode (ExecutionMode): The execution mode for the property.
            property_type (PropertyType): The type of the property.
            min_val (float): The minimum value of the property. Defaults to 0.0.
            max_val (float): The maximum value of the property. Defaults to 1.
            num_buckets (int): The number of buckets for the property. Defaults to 10.
            batch_size (int): The batch size for the property. Defaults to 1.
            dop (int): The degree of parallelism for the property. Defaults to 1.
            data_only_on_primary (bool): Whether the property data is only on the primary. Defaults to True.
        """
        return run_async_until_complete(
            self._add_property(
                property_name,
                setup_func,
                calc_func,
                execution_mode,
                property_type,
                min_val,
                max_val,
                num_buckets,
                batch_size,
                dop,
                data_only_on_primary,
            )
        )

    async def _add_property(
        self,
        property_name: str,
        setup_func: Callable,
        calc_func: Callable,
        execution_mode: ExecutionMode,
        property_type: "PropertyType",
        min_val: float = 0.0,
        max_val: float = 1.0,
        num_buckets: int = 10,
        batch_size: int = 1,
        dop: int = 1,
        data_only_on_primary: bool = True,
    ) -> None:
        """
        Asynchronously adds a property to the server.

        Args:
            property_name (str): The name of the property.
            setup_func (Callable): The setup function for the property.
            calc_func (Callable): The calculation function for the property.
            execution_mode (ExecutionMode): The execution mode for the property.
            property_type (PropertyType): The type of the property.
            min_val (float): The minimum value of the property. Defaults to 0.0.
            max_val (float): The maximum value of the property. Defaults to 1.
            num_buckets (int): The number of buckets for the property. Defaults to 10.
            batch_size (int): The batch size for the property. Defaults to 1.
            dop (int): The degree of parallelism for the property. Defaults to 1.
            data_only_on_primary (bool): Whether the property data is only on the primary. Defaults to True.
        """
        reader, writer = await self._connect_to_server()

        if reader is None or writer is None:
            return

        # Announce we want to add a property
        await write_int(int(ServerTask.ADD_PROPERTY), NUM_BYTES_FOR_IDENTIFIERS, writer)

        # Announce property name
        await write_utf8_string(property_name, NUM_BYTES_FOR_IDENTIFIERS, writer)

        # Announce setup function
        await write_pickeled_object(setup_func, NUM_BYTES_FOR_SIZES, writer)

        # Announce calculation function
        await write_pickeled_object(calc_func, NUM_BYTES_FOR_SIZES, writer)

        # Announce execution mode
        await write_int(execution_mode.value, NUM_BYTES_FOR_IDENTIFIERS, writer)

        # Announce property type
        await write_int(property_type.value, NUM_BYTES_FOR_IDENTIFIERS, writer)

        # Announce min value
        await write_float(min_val, writer)

        # Announce max value
        await write_float(max_val, writer)

        # Announce number of buckets
        await write_int(num_buckets, NUM_BYTES_FOR_IDENTIFIERS, writer)

        # Announce batch size
        await write_int(batch_size, NUM_BYTES_FOR_IDENTIFIERS, writer)

        # Announce degree of parallelism
        await write_int(dop, NUM_BYTES_FOR_IDENTIFIERS, writer)

        # Announce data only on primary
        await write_int(data_only_on_primary, NUM_BYTES_FOR_IDENTIFIERS, writer)
