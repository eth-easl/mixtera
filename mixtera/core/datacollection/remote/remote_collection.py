from typing import TYPE_CHECKING, Callable, Generator, Optional, Type

from loguru import logger
from mixtera.core.datacollection import MixteraDataCollection
from mixtera.core.datacollection.datasets import Dataset
from mixtera.core.datacollection.index import IndexType
from mixtera.core.processing.execution_mode import ExecutionMode
from mixtera.core.query import RemoteQueryResult
from mixtera.network.connection import ServerConnection

if TYPE_CHECKING:
    from mixtera.core.datacollection import PropertyType
    from mixtera.core.query import Query, QueryResult


class RemoteDataCollection(MixteraDataCollection):

    def __init__(self, host: str, port: int) -> None:
        self._server_connection = ServerConnection(host, port)
        self._host = host
        self._port = port

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

    def stream_query_results(
        self, query_result: "QueryResult", tunnel_via_server: bool = False
    ) -> Generator[str, None, None]:
        yield from MixteraDataCollection._stream_query_results(
            query_result, self._server_connection if tunnel_via_server else None
        )

    def is_remote(self) -> bool:
        return True

    def register_dataset(
        self,
        identifier: str,
        loc: str,
        dtype: Type[Dataset],
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
        raise NotImplementedError()
