import asyncio
import threading
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any, Callable, Generator, Optional, Type

from loguru import logger
from mixtera.core.datacollection import IndexType, MixteraDataCollection
from mixtera.core.datacollection.datasets import Dataset
from mixtera.core.filesystem import AbstractFilesystem
from mixtera.core.processing.execution_mode import ExecutionMode
from mixtera.network.connection import ServerConnection
from mixtera.network.server.server import ID_BYTES, SAMPLE_SIZE_BYTES
from mixtera.network.server_task import ServerTask
from mixtera.utils import run_async_until_complete
from mixtera.utils.network_utils import read_int, read_utf8_string, write_int, write_pickeled_object, write_utf8_string
from mixtera.core.query import  RemoteQueryResult

if TYPE_CHECKING:
    from mixtera.core.datacollection import PropertyType
    from mixtera.core.query import QueryResult, Query


class RemoteDataCollection(MixteraDataCollection):

    def __init__(self, host: str, port: int, prefetch_buffer_size: int) -> None:
        self._server_connection = ServerConnection(host, port)
        self._host = host
        self._port = port


    def stream_query_results(self, query_result: "QueryResult", tunnel_via_server: bool = False) -> Generator[str, None, None]:
        yield from MixteraDataCollection._stream_query_results(query_result, self._server_connection if tunnel_via_server else None)

    def is_remote(self) -> bool:
        return True

    def execute_query_at_server(self, query: "Query", chunk_size: int) -> RemoteQueryResult:
        if (query_id := self._server_connection.execute_query(query, chunk_size)) < 0:
            raise RuntimeError("Could not register query, got back invalid ID from server!")

        logger.info(f"Registered query with query id {query_id} for training {query.training_id}")

        return RemoteQueryResult(self._server_connection, query_id)


    def get_query_result(self, training_id: str) -> "RemoteQueryResult":
        logger.info(
            "Obtaining query id from server. This may take some time if primary node has not registered the query yet."
        )
        if (query_id := self._server_connection.get_query_id(training_id)) < 0:
            raise RuntimeError("Could not register query, got back invalid ID from server!")

        logger.info(f"Got query id {query_id} for training {training_id}")

        return RemoteQueryResult(self._server_connection, query_id)

    def get_samples_from_ranges(
        self, ranges_per_dataset_and_file: dict[int, dict[int, list[tuple[int, int]]]]
    ) -> Generator[str, None, None]:
        raise NotImplementedError(
            "Querying ranges from a RemoteDataCollection is currently not supported. Please run a query instead."
        )

    def register_dataset(
        self,
        identifier: str,
        loc: str,
        dtype: Type[Dataset],
        filesystem_t: Type[AbstractFilesystem],
        parsing_func: Callable[[str], str],
        metadata_parser_type: str,
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
