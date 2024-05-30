import asyncio
from typing import TYPE_CHECKING, Any, Callable, Generator, Iterable, Optional

from loguru import logger
from mixtera.network import NUM_BYTES_FOR_IDENTIFIERS, NUM_BYTES_FOR_SIZES
from mixtera.network.network_utils import (
    read_int,
    read_pickeled_object,
    read_utf8_string,
    write_int,
    write_pickeled_object,
    write_utf8_string,
)
from mixtera.network.server_task import ServerTask
from mixtera.utils import run_async_until_complete

if TYPE_CHECKING:
    from mixtera.core.datacollection.index import IndexType
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

        # Announce metadata
        await write_pickeled_object(mixture, NUM_BYTES_FOR_SIZES, writer)
        # await write_int(mixture, NUM_BYTES_FOR_IDENTIFIERS, writer)

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
    async def _get_next_result(self, job_id: str) -> Optional["IndexType"]:
        """
        Asynchronously retrieves the next result chunk of a query from the server.

        Args:
            job_id (str): The identifier of the job for which the next result chunk is requested.

        Returns:
            An IndexType object representing the next result chunk,
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

    def _stream_result_chunks(self, job_id: str) -> Generator["IndexType", None, None]:
        """
        Streams the result chunks of a query job from the server.

        Args:
            job_id (str): The identifier of the job whose result chunks are to be streamed.

        Yields:
            IndexType objects, each representing a chunk of the query results.
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
