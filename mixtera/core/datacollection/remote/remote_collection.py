import asyncio
import threading
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any, Callable, Generator, Optional, Type

from loguru import logger
from mixtera.core.datacollection import IndexType, MixteraDataCollection
from mixtera.core.datacollection.datasets import Dataset
from mixtera.core.processing.execution_mode import ExecutionMode
from mixtera.server.server import ID_BYTES, SAMPLE_SIZE_BYTES
from mixtera.server.server_task import ServerTask
from mixtera.utils import run_in_async_loop_and_return
from mixtera.utils.network_utils import read_int, read_utf8_string, write_int, write_pickeled_object, write_utf8_string

if TYPE_CHECKING:
    from mixtera.core.datacollection import PropertyType


class RemoteDataCollection(MixteraDataCollection):

    def __init__(self, host: str, port: int, prefetch_buffer_size: int) -> None:
        self._host = host
        self._port = port
        self._prefetch_buffer_size = prefetch_buffer_size
        self._data_queue = Queue(maxsize=prefetch_buffer_size)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._loop = None
        self._loop_started_event = threading.Event()  # Signal that the loop is running

        if self._prefetch_buffer_size < 1:
            raise RuntimeError(f"prefetch_buffer_size = {self._prefetch_buffer_size} < 1")

    def _start_async_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop_started_event.set()
        self._loop.run_forever()

    def _stop_async_loop(self):
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join()

        self._thread = None

    async def _connect_to_server(self) -> tuple[Optional[asyncio.StreamReader], Optional[asyncio.StreamWriter]]:
        try:
            logger.debug("Connecting to server.")
            reader, writer = await asyncio.wait_for(asyncio.open_connection(self._host, self._port), timeout=5.0)
        except asyncio.TimeoutError:
            logger.error(f"Connection to {self._host}:{self._port} timed out.")
            return None, None
        except Exception as e:
            logger.error(f"Failed to connect to {self._host}:{self._port}. Is the server running?: {e}")
            return None, None

        logger.debug("Connected!")

        return reader, writer

    async def _fetch_data_async(self, query_id: int, node_id: int, worker_id: int):
        reader, writer = await self._connect_to_server()

        if reader is None or writer is None:
            await self._loop.run_in_executor(None, self._data_queue.put, None)
            return

        # Announce we want to stream data
        await write_int(int(ServerTask.StreamData), ID_BYTES, writer)

        # Announce metadata
        await write_int(query_id, ID_BYTES, writer, drain=False)
        await write_int(node_id, ID_BYTES, writer, drain=False)
        await write_int(worker_id, ID_BYTES, writer)

        try:
            while not self._stop_event.is_set():
                # In the future, we might night custom logic using read_bytes here, if samples are not just strings
                if (sample := await read_utf8_string(SAMPLE_SIZE_BYTES, reader)) is not None:
                    await self._loop.run_in_executor(None, self._data_queue.put, sample)
                else:
                    break

        except Exception as e:
            logger.error(f"There was an exception in the asyncio event loop!: {e}")
            logger.exception(e)
        finally:
            try:
                logger.info("Closing writer.")
                writer.close()
                await asyncio.wait_for(writer.wait_closed(), timeout=5)
            except Exception as e:
                logger.error(
                    "Error while closing writer. This may be related to the previous error, "
                    + f"for example if the server crashed: {e}"
                )
                logger.exception(e)
            finally:
                logger.info("Connections closed.")
                await self._loop.run_in_executor(None, self._data_queue.put, None)  # Sentinel value

    def _start_fetching_data_for_query(self, query_id: int, node_id: int, worker_id: int) -> None:
        if self._thread is not None:
            # we might need multiple threads if RDCs are shared
            raise RuntimeError("RemoteDataCollection currently does not support multiple queries in parallel.")

        self._thread = threading.Thread(target=self._start_async_loop)
        self._thread.start()
        self._loop_started_event.wait()  # wait for the event loop to run
        asyncio.run_coroutine_threadsafe(self._fetch_data_async(query_id, node_id, worker_id), self._loop)

    def stream_query_results(self, query_id: int, node_id: int, worker_id: int) -> Generator[str, None, None]:
        self._start_fetching_data_for_query(query_id, node_id, worker_id)

        while not (self._stop_event.is_set() and self._data_queue.empty()):
            try:
                data_chunk = self._data_queue.get(timeout=1.0)
            except Empty:
                continue  # while loop breaks by condition

            if data_chunk is None:
                # Set the event for shutting down the coroutine loop and wait for it to finish
                self._stop_event.set()
                self._stop_async_loop()
                continue  # outer while loop ends afterwards

            yield data_chunk

    async def _register_query(self, query: Any, training_id: str, num_nodes: int, num_workers_per_node: int) -> int:
        reader, writer = await self._connect_to_server()

        if reader is None or writer is None:
            return -1

        # Announce we want to register a query
        await write_int(int(ServerTask.RegisterQuery), ID_BYTES, writer)

        # Announce training ID
        await write_utf8_string(training_id, SAMPLE_SIZE_BYTES, writer)

        # Announce metadata
        await write_int(num_nodes, ID_BYTES, writer, drain=False)
        await write_int(num_workers_per_node, ID_BYTES, writer)

        # Announce query
        await write_pickeled_object(query, SAMPLE_SIZE_BYTES, writer)

        query_id = await read_int(ID_BYTES, reader)
        logger.debug(f"Got query id {query_id} from server.")
        return query_id

    # TODO(MaxiBoether): Change Query type accordingly
    def register_query(self, query: Any, training_id: str, num_nodes: int, num_workers_per_node: int) -> int:
        if (
            query_id := run_in_async_loop_and_return(
                self._register_query(query, training_id, num_nodes, num_workers_per_node)
            )
        ) < 0:
            raise RuntimeError("Could not register query, got back invalid ID from server!")

        logger.info(f"Registered query with query id {query_id} for training {training_id}")

        return query_id

    async def _get_query_id(self, training_id: str) -> int:
        reader, writer = await self._connect_to_server()

        if reader is None or writer is None:
            return -1

        # Announce we want to get the query id
        await write_int(int(ServerTask.GetQueryId), ID_BYTES, writer)

        # Announce training ID
        await write_utf8_string(training_id, SAMPLE_SIZE_BYTES, writer)
        query_id = await read_int(ID_BYTES, reader, timeout=30)
        logger.debug(f"Got query id {query_id} from server.")
        return query_id

    def get_query_id(self, training_id: str) -> int:
        logger.info(
            "Obtaining query id from server. This may take some time if primary node has not registered the query yet."
        )
        if (query_id := run_in_async_loop_and_return(self._get_query_id(training_id))) < 0:
            raise RuntimeError("Could not register query, got back invalid ID from server!")

        logger.info(f"Got query id {query_id} for training {training_id}")

        return query_id

    def get_samples_from_ranges(
        self, ranges_per_dataset_and_file: dict[int, dict[int, list[tuple[int, int]]]]
    ) -> Generator[str, None, None]:
        raise NotImplementedError(
            "Querying ranges from a RemoteDataCollection is currently not supported. Please run a query insetad."
        )

    def register_dataset(
        self, identifier: str, loc: str, dtype: Type[Dataset], parsing_func: Callable[[str], str]
    ) -> bool:
        raise NotImplementedError()

    def check_dataset_exists(self, identifier: str) -> bool:
        raise NotImplementedError()

    def list_datasets(self) -> list[str]:
        raise NotImplementedError()

    def remove_dataset(self, identifier: str) -> bool:
        raise NotImplementedError()

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
        raise NotImplementedError()

    def get_index(self, property_name: Optional[str] = None) -> IndexType:
        """
        This function returns the index of the MixteraDataCollection.

        Args:
            property_name (Optional[str], optional): The name of the property to query.
                If not provided, all properties are returned.
        """
        raise NotImplementedError()
